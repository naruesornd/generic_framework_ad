import pandas as pd

# =============================================
#             Cycle Analysis
# =============================================

import pandas as pd
import numpy as np

class CycleProcessor:
    def __init__(self, column_name, df, min_cycle_length=70, threshold=0.05):
        self.cycles = []
        self.min_cycle_length = min_cycle_length
        self.threshold = threshold
        self.column_name = column_name
        self.df = df

    def identify_cycles(self):
        if self.df is not None:
            target_feature = self.df[self.column_name].values
            diffs = np.abs(np.diff(target_feature))
            change_points = np.where(diffs > self.threshold)[0] + 1

            cycles = []
            start = 0
            for cp in change_points:
                if cp - start >= self.min_cycle_length:
                    cycles.append((start, cp))
                    start = cp
                else:
                    start = cp
            if len(target_feature) - start >= self.min_cycle_length:
                cycles.append((start, len(target_feature)))
            self.cycles = cycles
        else:
            print('No data loaded')
    def assign_cycle_features(self, cycle_offset=0):
        if self.df is not None:
            self.df["cycle_id"] = -1
            self.df["cycle_time"] = -1

            for i, (start, end) in enumerate(self.cycles):
                self.df.loc[self.df.index[start:end], "cycle_id"] = i + cycle_offset
                self.df.loc[self.df.index[start:end], "cycle_time"] = np.arange(end - start)
            return self.df
        else:
            print('No data loaded')
            return None

    def trim_cycle_boundaries(self, n_trim=1):
        """
        Removes the first and last n_trim rows of each assigned cycle to exclude
        backwash/cleaning transition drops from model training.
        Only call this AFTER assign_cycle_features().

        Args:
            n_trim: Number of timestamps to trim from each cycle boundary (default=1).

        Returns:
            Trimmed DataFrame with reset index. Also updates self.df in place.
        """
        if self.df is None:
            print('No data loaded')
            return None

        if "cycle_id" not in self.df.columns:
            print("❌ Run assign_cycle_features() before trimming.")
            return None

        original_len = len(self.df)

        # Collect boundary indices to drop for each valid cycle
        drop_indices = []
        for cycle_id, group in self.df.groupby("cycle_id"):
            if cycle_id == -1:          # skip unassigned rows
                continue
            if len(group) <= n_trim * 2:  # skip cycles too short to trim
                print(f"⚠️  Cycle {cycle_id} too short to trim (len={len(group)}), skipping.")
                continue
            drop_indices.extend(group.index[:n_trim].tolist())  # head
            drop_indices.extend(group.index[-n_trim:].tolist()) # tail

        self.df = self.df.drop(index=drop_indices).reset_index(drop=True)

        n_dropped = original_len - len(self.df)
        n_cycles  = self.df[self.df["cycle_id"] != -1]["cycle_id"].nunique()
        print(f"✅ Trimmed {n_dropped} boundary rows "
              f"(±{n_trim} per cycle) across {n_cycles} cycles.")

        return self.df
        
    def export_files(self, file_path):
        if self.df is not None:
            self.df.reset_index(drop=False)
            self.df.to_csv(file_path, index=False)
        else:
            print('No data loaded')


    def export_cycle_report(self, output_path='cycle_report.csv'):
        """
        Exports a summary report of each cycle with statistics.
        
        Args:
            output_path: Path to save the CSV report (e.g., 'cycle_report.csv')
        
        Returns:
            DataFrame with cycle statistics
        """
        if self.df is None:
            print('No data loaded')
            return None
        
        if "cycle_id" not in self.df.columns:
            print("❌ Run assign_cycle_features() before exporting report.")
            return None
        
        report_data = []
        
        for cycle_id, group in self.df[self.df['cycle_id'] != -1].groupby('cycle_id'):
            target_vals = group[self.column_name]
            
            report_data.append({
                'cycle_id': int(cycle_id),
                'start_timestamp': group['timestamp'].iloc[0],
                'end_timestamp': group['timestamp'].iloc[-1],
                'duration_rows': len(group),
                'target_min': target_vals.min(),
                'target_max': target_vals.max(),
                'target_mean': target_vals.mean(),
                'target_std': target_vals.std(),
            })
        
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(output_path, index=False)
        
        print(f"✅ Cycle report exported to {output_path}")
        print(f"\n{report_df.to_string()}")
        
        return report_df



# --------------------------------------------------
#               NOT USE
# --------------------------------------------------

# class CycleProcessor:
#     def __init__(self, dp):
#         self.dp = dp
#         self.df = dp.df

#     def extract_hybrid_cycles(self, time_col='timestamp', max_gap_hours=1.5, trim_start_hours=1, trim_end_hours=1, min_cycle_hours=5):
#         """
#         Identifies RO cycles, trims transients, and RETURNS the clean data
#         without modifying the original DataProcessor dataframe.
#         """
#         print("--- ⚙️ Extracting Hybrid Cycles (NaN & Time-Gap Aware) ---")

#         # We work strictly on a copy
#         df = self.df.copy()
#         df[time_col] = pd.to_datetime(df[time_col])
#         df = df.sort_values(by=time_col).reset_index(drop=True)

#         # 1. Define what "Valid" data looks like
#         core_sensors = ['PermeateFlow', 'FeedPressure', 'PermeateConductivity']
#         is_valid = df[core_sensors].notna().all(axis=1)

#         # 2. Check for Time Gaps (e.g., sudden jumps of > 1.5 hours)
#         time_gaps = df[time_col].diff() > pd.Timedelta(hours=max_gap_hours)

#         # 3. TRIGGER LOGIC: When does a NEW cycle start?
#         prev_valid = is_valid.shift(1, fill_value=False)
#         new_cycle_starts = is_valid & (~prev_valid | time_gaps)

#         # 4. Assign IDs dynamically using a cumulative sum
#         df['Cycle_ID'] = new_cycle_starts.cumsum()

#         # Force offline/NaN rows to have Cycle_ID = 0
#         df.loc[~is_valid, 'Cycle_ID'] = 0

#         # Add Cycle_Time to full preserved df (NaN for offline/gap rows)
#         active_mask = df['Cycle_ID'] > 0
#         df['Cycle_Time'] = float('nan')
#         df.loc[active_mask, 'Cycle_Time'] = (
#             df[active_mask].groupby('Cycle_ID')[time_col]
#             .transform(lambda x: (x - x.min()).dt.total_seconds() / 3600.0)
#         )

#         # =========================================================
#         # Phase 2: Process, Trim, and Summarize
#         # =========================================================
#         active_df = df[df['Cycle_ID'] > 0].copy()
#         steady_state_data = []
#         summary_data = []

#         previous_end_time = None

#         for cycle_id, cycle_data in active_df.groupby('Cycle_ID'):

#             start_time = cycle_data[time_col].min()
#             end_time = cycle_data[time_col].max()
#             total_hours = (end_time - start_time).total_seconds() / 3600.0

#             # Calculate Downtime
#             downtime_hours = 0.0
#             if previous_end_time is not None:
#                 downtime_hours = (start_time - previous_end_time).total_seconds() / 3600.0
            
#             # Filter out short cycles
#             if total_hours < min_cycle_hours or total_hours <= (trim_start_hours + trim_end_hours):
#                 continue
            
#             previous_end_time = end_time

#             # Add Cycle Age
#             cycle_data_copy = cycle_data.copy()
#             cycle_data_copy['Cycle_Time'] = (cycle_data_copy[time_col] - start_time).dt.total_seconds() / 3600.0

#             # Trim Transients
#             safe_start = start_time + pd.Timedelta(hours=trim_start_hours)
#             safe_end = end_time - pd.Timedelta(hours=trim_end_hours)
#             steady_cycle = cycle_data_copy[(cycle_data_copy[time_col] >= safe_start) & (cycle_data_copy[time_col] <= safe_end)]

#             steady_state_data.append(steady_cycle)

#             summary_data.append({
#                 'Cycle_ID': cycle_id,
#                 'Start_Time': start_time,
#                 'End_Time': end_time,
#                 'Raw_Duration_Hours': total_hours,
#                 'Steady_State_Hours': len(steady_cycle),
#                 'Downtime_Before_Start_Hours': downtime_hours
#             })

#         if len(steady_state_data) == 0:
#             print("⚠️ No valid steady-state cycles found.")
#             return pd.DataFrame(), pd.DataFrame(), df

#         final_clean_df = pd.concat(steady_state_data).reset_index(drop=True)
#         summary_df = pd.DataFrame(summary_data)

#         print(f"✅ Extracted {len(summary_df)} high-quality hybrid cycles.")

#         # Returns three dataframes:
#         #   final_clean_df — trimmed steady-state rows only (for model calibration)
#         #   summary_df     — one row per cycle with duration/downtime stats
#         #   df             — full original data preserved, with Cycle_ID and Cycle_Time added
#         return final_clean_df, summary_df, df

#     def detect_cycles(self, 
#         dp_col         = "DifferentialPressure",
#         smooth_window  = 24,
#         drop_threshold = -3.0,
#         merge_days     = 3,
#         min_cycle_days = 5,
#         time_col = "timestamp"
#     ):
#         """Adds cycle detection to the dataframe"""
#         df = self.df.copy()  # Use self.df instead of parameter
#         df[time_col] = pd.to_datetime(df[time_col])
#         df = df.sort_values(time_col).reset_index(drop=True)

#         # Step 1 — smooth
#         dp_smooth = df[dp_col].rolling(smooth_window, min_periods=4, center=True).mean()
#         # dp_smooth = df[dp_col]

#         # Step 2 — daily change
#         daily = (
#             df.assign(_dp_s=dp_smooth)
#             .set_index(time_col)["_dp_s"]
#             .resample("D").mean()
#             .reset_index()
#         )
#         daily.columns = ["date", "dp"]
#         daily["chg"] = daily["dp"].diff()

#         # Step 3 — flag big drops
#         flagged = daily[daily["chg"] < drop_threshold].sort_values("date").reset_index(drop=True)

#         if flagged.empty:
#             df["cycle_id"] = 1
#             df["cycle_time_hours"] = (
#                 (df[time_col] - df[time_col].iloc[0]).dt.total_seconds() / 3600
#             ).round(2)
#             self.df = df  # Update the dataframe
#             return df

#         # Step 4 — merge nearby flags
#         events = []
#         best_dp, best_date = flagged.iloc[0]["dp"], flagged.iloc[0]["date"]
#         for i in range(1, len(flagged)):
#             row, prev = flagged.iloc[i], flagged.iloc[i - 1]
#             if (row["date"] - prev["date"]).days <= merge_days:
#                 if row["dp"] < best_dp:
#                     best_dp, best_date = row["dp"], row["date"]
#             else:
#                 events.append(best_date)
#                 best_dp, best_date = row["dp"], row["date"]
#         events.append(best_date)

#         # Step 5 — enforce minimum cycle length
#         filtered = [events[0]]
#         for e in events[1:]:
#             if (e - filtered[-1]).days >= min_cycle_days:
#                 filtered.append(e)

#         # Step 6 — label every row
#         df["cycle_id"] = 0
#         df["cycle_time"] = 0.0
#         starts = [df["timestamp"].iloc[0]] + [pd.Timestamp(e) for e in filtered]

#         for i, start in enumerate(starts):
#             end = (starts[i + 1] if i + 1 < len(starts)
#                 else df["timestamp"].iloc[-1] + pd.Timedelta(hours=1))
#             mask = (df[time_col] >= start) & (df[time_col] < end)
#             df.loc[mask, "cycle_id"] = i + 1
#             df.loc[mask, "cycle_time_hours"] = (
#                 (df.loc[mask, time_col] - start).dt.total_seconds() / 3600
#             ).round(2)

#         self.df = df  # Update the dataframe
#         return df
    
#     def detect_cycles_hourly(self, 
#                         dp_col="DifferentialPressure",
#                         smooth_window=24,
#                         drop_threshold=-1.5,  # Lower threshold for hourly
#                         merge_hours=6,        # Merge flags within 6 hours
#                         min_cycle_hours=120,  # Minimum 5 days between cleanings
#                         time_col="timestamp"):
#         """
#         Cycle detection based on HOURLY changes (better for hourly data)
#         """
#         df = self.df.copy()
#         df[time_col] = pd.to_datetime(df[time_col])
#         df = df.sort_values(time_col).reset_index(drop=True)
        
#         # Step 1: Smooth (optional)
#         if smooth_window > 1:
#             dp_smooth = df[dp_col].rolling(smooth_window, min_periods=0, center=True).mean()
#         else:
#             dp_smooth = df[dp_col]
        
#         # Step 2: Calculate HOURLY changes
#         df['dp_smooth'] = dp_smooth
#         df['hourly_change'] = df['dp_smooth'].diff()
        
#         # Step 3: Flag big HOURLY drops
#         # Look for sudden drops within a single hour
#         flagged_hours = df[df['hourly_change'] < drop_threshold].copy()
        
#         if flagged_hours.empty:
#             df["cycle_id"] = 1
#             df["cycle_time_hours"] = (
#                 (df[time_col] - df[time_col].iloc[0]).dt.total_seconds() / 3600
#             ).round(2)
#             self.df = df
#             return df
        
#         # Step 4: Merge nearby flags (within merge_hours)
#         events = []
#         current_event_time = flagged_hours.iloc[0][time_col]
#         current_event_dp = flagged_hours.iloc[0][dp_col]
        
#         for idx in range(1, len(flagged_hours)):
#             current_time = flagged_hours.iloc[idx][time_col]
#             time_diff = (current_time - current_event_time).total_seconds() / 3600
            
#             if time_diff <= merge_hours:
#                 # Same cleaning event, keep the lowest DP
#                 if flagged_hours.iloc[idx][dp_col] < current_event_dp:
#                     current_event_dp = flagged_hours.iloc[idx][dp_col]
#                     current_event_time = current_time
#             else:
#                 # Different cleaning event
#                 events.append(current_event_time)
#                 current_event_time = current_time
#                 current_event_dp = flagged_hours.iloc[idx][dp_col]
        
#         events.append(current_event_time)
        
#         # Step 5: Enforce minimum cycle length
#         filtered = [events[0]]
#         for e in events[1:]:
#             cycle_hours = (e - filtered[-1]).total_seconds() / 3600
#             if cycle_hours >= min_cycle_hours:
#                 filtered.append(e)
        
#         # Step 6: Label cycles
#         df["cycle_id"] = 0
#         df["cycle_time_hours"] = 0.0
#         starts = [df[time_col].iloc[0]] + filtered
        
#         for i, start in enumerate(starts):
#             end = starts[i+1] if i+1 < len(starts) else df[time_col].iloc[-1] + pd.Timedelta(hours=1)
#             mask = (df[time_col] >= start) & (df[time_col] < end)
#             df.loc[mask, "cycle_id"] = i + 1
#             df.loc[mask, "cycle_time_hours"] = (
#                 (df.loc[mask, time_col] - start).dt.total_seconds() / 3600
#             ).round(2)
        
#         self.df = df
#         return df
    

