import pandas as pd
import numpy as np
from itertools import combinations
from data_processor import DataProcessor
from datetime import datetime
from sklearn.linear_model import LinearRegression

class FeatureEngineering:
    def __init__(self,dp:DataProcessor):
        self.dp = dp
        self.df = dp.df
        self.cross_features = []
        # self.features = self.dp.list_columns(print_columns=False)
        # self.features = [f for f in self.features if f not in ['timestamp']]
        self.features = [f for f in self.df.columns if f not in ['timestamp']]
                
    def generate_cross_features(self, drop_features=[]):
        features = [f for f in self.features if f not in drop_features]
        new_cols = {}
        for f1,f2 in combinations(features,2):
            new_cols[f1 + '_x_' + f2] = self.df[f1] * self.df[f2]
        self.cross_features = new_cols.keys()
        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)
        return self.cross_features
    
    
    def cycle_time_engineer(self):
        new_cols = {
        'hour_sin': np.sin(2 * np.pi * self.df['timestamp'].dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * self.df['timestamp'].dt.hour / 24),
        'day_sin':  np.sin(2 * np.pi * self.df['timestamp'].dt.dayofyear / 365),
        'day_cos':  np.cos(2 * np.pi * self.df['timestamp'].dt.dayofyear / 365),
        }
        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

        
    def lag_engineer(self, lags=(1, 2, 3, 6, 12), drop_features=[], mode="cycle"):
        """
        Create lag features.

        mode = "cycle"  -> lags computed within each cycle_id (no cross-cycle leakage)
        mode = "global" -> lags computed across entire dataframe (continuous)
        mode = "both"   -> create both, with different names:
                           f + "_cycle_lag_k" and f + "_global_lag_k"
        """
        if mode not in ["cycle", "global", "both"]:
            raise ValueError("mode must be 'cycle', 'global', or 'both'")

        new_cols = {}

        # we don't lag cycle_id itself, but allow cycle_time to be lagged if present
        base_features = [f for f in self.features if f not in drop_features]

        has_cycle = "cycle_id" in self.df.columns

        for f in base_features:
            for lag in lags:
                if mode in ["cycle", "both"] and has_cycle:
                    # lag inside each cycle
                    col_name = f"{f}_cycle_lag_{lag}" if mode == "both" else f"{f}_lag_{lag}"
                    new_cols[col_name] = self.df.groupby("cycle_id")[f].shift(lag)

                if mode in ["global", "both"]:
                    # lag over the whole dataframe (no groupby)
                    col_name = f"{f}_global_lag_{lag}" if mode == "both" else f"{f}_lag_{lag}"
                    new_cols[col_name] = self.df[f].shift(lag)

        # add new lag columns
        self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)
        
    def rolling_mean_engineer(self):
        new_cols = {}
        for f in self.features:
            for window in [1,2,3,6,12]:
                new_cols[f + '_rolling_mean_' + str(window)] = self.df[f].rolling(window).mean()
        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

    def rolling_std_engineer(self):
        new_cols = {}
        for f in self.features:
            for window in [1,2,3,6,12]:
                new_cols[f + '_rolling_std_' + str(window)] = self.df[f].rolling(window).std()
        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

    def add_constant_features(self, const_dict):
        """
        Add multiple constant features at once.
        const_dict: dict like {'FeedFlow_setpoint': 300, 'OtherSetpoint': 5.0}
        """
        for name, value in const_dict.items():
            self.df[name] = value
            if name not in self.features and name != 'timestamp':
                self.features.append(name)
        # keep dp.df in sync
        self.dp.df = self.df
        # return list(const_dict.keys())

    def add_salt_rejection(self,
                           feed_col='FeedConductivity',
                           permeate_col='PermeateConductivity',
                           frac_col='SaltRejection_fraction',
                           pct_col='SaltRejection_percentage'):
        """
        Add salt rejection features based on feed and permeate conductivity.

        R = 1 - (C_permeate / C_feed)

        - frac_col: rejection as a fraction 0–1
        - pct_col : rejection as percentage 0–100
        """

        if feed_col not in self.df.columns:
            raise ValueError(f"{feed_col} not found in df.columns")
        if permeate_col not in self.df.columns:
            raise ValueError(f"{permeate_col} not found in df.columns")

        feed = self.df[feed_col].astype(float)
        perm = self.df[permeate_col].astype(float)

        # avoid division by zero / NaN
        valid_mask = (feed > 0) & feed.notna() & perm.notna()

        frac = np.full(len(self.df), np.nan, dtype=float)
        frac[valid_mask] = 1.0 - (perm[valid_mask] / feed[valid_mask])

        self.df[frac_col] = frac
        self.df[pct_col] = frac * 100.0

        # keep dp.df in sync
        self.dp.df = self.df

        # update feature list
        for c in [frac_col, pct_col]:
            if c not in self.features:
                self.features.append(c)

        return frac_col, pct_col
    



################# PHYSICS BASED FEATURES ######################
###############################################################
###############################################################

class PhysicsBasedFeatures:
    def __init__(self,dp:DataProcessor):
        
        self.dp = dp
        self.df = dp.df
        self.features = [f for f in self.df.columns if f not in ['timestamp']]

    def differential_pressure(self, flow_col, temp_col, calibration_col=None,k=None, beta=1.7):
        """
        Calculates the theoretical Differential Pressure (friction loss) based on 
        Flow and Temperature (viscosity).
    
        Parameters:
        - self: The pandas DataFrame containing the data.
        - flow_col: Name of the column with Feed Flow (or Average Flow) rate.
        - temp_col: Name of the column with Feed Temperature (in Celsius).
        - k: Friction coefficient (if None, it automtically calibrates to the first 100 rows).
        - beta: Flow exponent (typically 1.5 to 2.0 for RO spacers).
        
        Returns:
        - numpy array of Predicted Delta P
        """
        # 1. Extract Vectors
        Q = self.df[flow_col].values
        T = self.df[temp_col].values
    
        # 2. Calculate Viscosity Correction Factor (Empirical approximation for water)
        # As Temp rises, Viscosity drops, so Friction drops.
        # Reference: approx 2.4% drop in viscosity per degree C near 25C.
        mu_correction = np.exp(-0.024 * (T - 25.0))
        
        # 3. The Physics Equation: dP ~ k * mu * Q^beta
        # We calculate the shape of the curve first (Unscaled)
        dP_shape = mu_correction * (Q ** beta)
        
        # 4. Calibration Logic
        if k is None:
            if calibration_col:
                # --- FIX 2: Real Auto-Calibration ---
                # We use the first 100 points (Clean Baseline) to find k
                # Filter to avoid ZeroDivision if flow is 0
                subset_shape = dP_shape[:100]
                subset_actual = self.df[calibration_col].iloc[:100].values
                
                # Calculate k that maps Shape -> Actual
                # Avoid dividing by zero using a mask or simple mean ratio
                valid_idx = subset_shape > 0.001
                k = np.mean(subset_actual[valid_idx]) / np.mean(subset_shape[valid_idx])
                print(f"Auto-calibrated k: {k:.4f}")
            else:
                print("Warning: No 'k' or 'calibration_col' provided. Returning unscaled shape.")
                k = 1.0
            
        dP_predicted = k * dP_shape
        
        return dP_predicted

    def estimate_permeate_pressure(self, 
                                   feed_press_col, 
                                   feed_cond_col, 
                                   physics_dp_col, 
                                   permeate_flow_col,
                                   actual_perm_press_col=None, 
                                   membrane_area_m2=37.0):
        """
        Back-solves the Solution-Diffusion model to estimate Permeate Pressure.
        
        Equation: P_perm = P_average_hydraulic - Osmotic_Pressure - (Flux / Permeability)
        
        Parameters:
        - feed_press_col: Column for Feed Pressure (Bar).
        - feed_cond_col: Column for Feed Conductivity (uS/cm).
        - physics_dp_col: The column created by the previous function (Physics Predicted DP).
        - permeate_flow_col: Column for Permeate Flow (m3/h).
        - actual_perm_press_col: (Optional) Used only for auto-calibration of Permeability (A).
        - membrane_area_m2: Total active membrane area (default 37.0 for a standard 8-inch element).
        """
        
        # 1. Calculate Average Hydraulic Pressure (Using Physics Delta P)
        # The pump pushes 'P_feed', but friction 'physics_dp' eats some of it.
        # P_avg is the pressure sitting in the middle of the element.
        P_feed = self.df[feed_press_col]
        dP_phys = self.df[physics_dp_col]
        P_avg = P_feed - (dP_phys / 2.0)
        
        # 2. Calculate Osmotic Pressure (Pi)
        # We assume a linear relationship for brackish/seawater.
        # Approx: 1000 uS/cm ~= 0.7 Bar Osmotic Pressure (Standard approximation)
        # Factor = 0.0007
        cond = self.df[feed_cond_col]
        Pi = 0.0007 * cond 
        
        # 3. Calculate Flux (J)
        # Flux is speed of water per unit area (m/h)
        Q_perm = self.df[permeate_flow_col]
        Flux = Q_perm / membrane_area_m2
        
        # 4. Calibrate Membrane Permeability (A)
        # We need to know how "porous" the membrane is when clean.
        # A = Flux / Net_Driving_Pressure
        if actual_perm_press_col:
            # Take clean baseline (first 100 rows)
            P_perm_actual = self.df[actual_perm_press_col].iloc[:100]
            
            # NDP = (Hydraulic_Diff) - (Osmotic_Diff)
            # NDP = (P_avg - P_perm) - Pi
            NDP_clean = (P_avg.iloc[:100] - P_perm_actual) - Pi.iloc[:100]
            
            # Calculate A (ignore zeros/noise)
            valid = NDP_clean > 0.1
            A_k = np.mean(Flux.iloc[:100][valid] / NDP_clean[valid])
            print(f"Auto-calibrated Permeability (A): {A_k:.4f} (m/h/bar)")
        else:
            # Default fallback if no calibration data exists (Typical for BWRO)
            print("Warning: No actual pressure for calibration. Using default A=1.0")
            A_k = 1.0
            
        # 5. Solve for Theoretical Permeate Pressure
        # The 'Unified Equation' rearranged:
        # P_perm = P_avg - Pi - (Flux / A)
        
        P_perm_pred = P_avg - Pi - (Flux / A_k)
        
        return P_perm_pred    
    
    def predict_pressures(self, 
                          flow_col, 
                          temp_col, 
                          cond_col, 
                          perm_flow_col, 
                          physics_dp_col,
                          perm_press_col=None,
                          membrane_area_m2=37.0, # Default: 6 elements
                          calibration_feed_col=None):
        """
        Calculates theoretical Feed and Concentrate pressures using the Solution-Diffusion Model.
        
        Returns a DataFrame with two new columns: 'Physics_Feed_Pressure', 'Physics_Conc_Pressure'
        """
        
        # 1. Gather Vectors
        Q_perm = self.df[perm_flow_col].values
        T = self.df[temp_col].values
        Cond = self.df[cond_col].values
        dP_phys = self.df[physics_dp_col].values
        
        # Handle Permeate Pressure (If sensor exists, use it; else assume 0)
        if perm_press_col:
            P_perm = self.df[perm_press_col].values
        else:
            P_perm = np.zeros(len(self.df))

        # 2. Calculate Osmotic Pressure (Pi)
        # Rule of Thumb: 1000 uS/cm ~= 0.7 bar (approx 0.0007 bar/uS)
        # We adjust for concentration polarization (roughly 10% higher at surface)
        Pi = (Cond * 0.0007) * 1.1 

        # 3. Calculate Temperature Correction Factor (TCF) for Permeability
        # Standard logic: Permeability increases ~3% per degree C
        TCF = np.exp(0.03 * (T - 25.0))

        # 4. Calculate Flux (J)
        Flux = Q_perm / membrane_area_m2

        # 5. Calibrate Reference Permeability (A_25)
        # We need to find the "Clean A" at 25C from the first 100 rows
        if calibration_feed_col:
            P_feed_act = self.df[calibration_feed_col].iloc[:100].values
            
            # Rearrange equation to solve for A:
            # P_feed = (J / (A_25 * TCF)) + Pi + P_perm + 0.5*dP
            # A_25 = J / [ (P_feed - Pi - P_perm - 0.5*dP) * TCF ]
            
            numerator = Flux[:100]
            # Net Driving Pressure
            denominator_pressure = (P_feed_act - Pi[:100] - P_perm[:100] - 0.5 * dP_phys[:100])
            denominator = denominator_pressure * TCF[:100]
            
            # Avoid division by zero
            valid = denominator > 0.1
            A_25 = np.mean(numerator[valid] / denominator[valid])
            print(f"Auto-calibrated Reference Permeability (A_25): {A_25:.5f}")
        else:
            print("Warning: No Feed Pressure for calibration. Using generic default.")
            A_25 = 1.0  # Generic placeholder

        # 6. Calculate Theoretical Feed Pressure
        # P_feed = (J / A_t) + Pi + P_perm + 0.5*dP
        Resistance_Term = Flux / (A_25 * TCF)
        
        P_feed_pred = Resistance_Term + Pi + P_perm + (0.5 * dP_phys)
        
        # 7. Calculate Theoretical Concentrate Pressure
        # P_conc = P_feed - dP_phys
        P_conc_pred = P_feed_pred - dP_phys
        
        return pd.DataFrame({
            'Physics_Feed_Pressure': P_feed_pred,
            'Physics_Conc_Pressure': P_conc_pred
        })


    def add_all_physics_features(self, 
                                 feed_temp_col='FeedTemperature',
                                 feed_cond_col='FeedConductivity',
                                 feed_press_col='FeedPressure',
                                 diff_press_col='DifferentialPressure',
                                 perm_press_col='PermeatePressure',
                                 perm_flow_col='PermeateFlow',
                                 perm_cond_col='PermeateConductivity'):
        """
        Generates ALL physics targets (Flow, Conductivity, Pressure) and their Residuals.
        Updates self.df directly.
        """
        print("--- Generating Physics Models for All Parameters ---")

        # 1. Gather Vectors (Numpy for speed)
        # We extract all necessary columns at once to keep the code clean
        T = self.df[feed_temp_col].values
        Cond_Feed = self.df[feed_cond_col].values
        P_feed = self.df[feed_press_col].values
        dP = self.df[diff_press_col].values
        P_perm = self.df[perm_press_col].values
        Q_perm = self.df[perm_flow_col].values
        Cond_Perm = self.df[perm_cond_col].values

        # 2. Global Constants
        # Temperature Correction Factor (ASTM D4516)
        # Standard logic: exp(0.027 * (T - 25))
        TCF = np.exp(0.027 * (T - 25.0))
        self.df['TCF'] = TCF
        
        # Osmotic Pressure Estimation
        # Approx: 0.0004 bar per uS/cm
        Pi_Feed = 0.0004 * Cond_Feed
        self.df['Osmotic_Pressure'] = Pi_Feed

        # ---------------------------------------------------------
        # 3. PERMEATE FLOW MODEL (Solution-Diffusion)
        # ---------------------------------------------------------
        # P_avg = FeedPress - (DP / 2)
        P_avg = P_feed - (dP / 2.0)
        
        # Net Driving Pressure (NDP) = P_avg - P_perm - Osmotic
        NDP = P_avg - P_perm - Pi_Feed
        self.df['P_avg'] = P_avg
        self.df['NDP'] = NDP
        
        # Calibrate A (Water Permeability) using first 500 rows
        # A = Q / (TCF * NDP)
        # Filter for valid NDP > 1.0 to avoid noise
        valid_calib = (NDP[:500] > 1.0)
        
        numerator = Q_perm[:500][valid_calib]
        denominator = TCF[:500][valid_calib] * NDP[:500][valid_calib]
        
        A_coeff = np.median(numerator / denominator)
        print(f"Calibrated Water Permeability (A): {A_coeff:.4f}")
        
        # Calculate Physics Flow
        Q_phys = A_coeff * TCF * NDP
        self.df['Physics_PermeateFlow'] = Q_phys
        self.df['Residual_PermeateFlow'] = Q_perm - Q_phys

        # ---------------------------------------------------------
        # 4. PERMEATE CONDUCTIVITY MODEL (Diffusion)
        # ---------------------------------------------------------
        # Salt Flux = Q_perm * C_perm
        # Driving Conc = C_feed * TCF
        Salt_Flux_Actual = Q_perm[:500] * Cond_Perm[:500]
        Driving_Conc = Cond_Feed[:500] * TCF[:500]
        
        # Calibrate B (Salt Permeability)
        # B = Salt_Flux / Driving_Conc
        # Avoid division by zero
        valid_salt = Driving_Conc > 0.1
        B_coeff = np.median(Salt_Flux_Actual[valid_salt] / Driving_Conc[valid_salt])
        print(f"Calibrated Salt Permeability (B): {B_coeff:.4f}")
        
        # Calculate Physics Conductivity
        # C_phys = (B * C_feed * TCF) / Q_phys
        # Use a safe flow (replace 0 with 0.1) to avoid Inf
        Q_phys_safe = np.where(Q_phys == 0, 0.1, Q_phys)
        
        Salt_Flux_Phys = B_coeff * Cond_Feed * TCF
        Cond_Phys = Salt_Flux_Phys / Q_phys_safe
        
        self.df['Physics_PermeateConductivity'] = Cond_Phys
        self.df['Residual_PermeateConductivity'] = Cond_Perm - Cond_Phys

        # ---------------------------------------------------------
        # 5. PERMEATE PRESSURE MODEL (Hydraulic)
        # ---------------------------------------------------------
        # Simple Linear Regression: P_perm ~ k * Q_perm + c
        # Calibrate on first 500 rows
        from sklearn.linear_model import LinearRegression
        
        X_calib = Q_perm[:500].reshape(-1, 1)
        y_calib = P_perm[:500]
        
        reg = LinearRegression()
        reg.fit(X_calib, y_calib)
        
        # Predict
        P_phys = reg.predict(Q_perm.reshape(-1, 1))
        
        self.df['Physics_PermeatePressure'] = P_phys
        self.df['Residual_PermeatePressure'] = P_perm - P_phys
        
        return self.df
        



# import pandas as pd
# import numpy as np
# from itertools import combinations
# from data_processor import DataProcessor
# from datetime import datetime
# class FeatureEngineering:
#     def __init__(self,dp:DataProcessor):
#         self.dp = dp
#         self.df = dp.df
#         self.cross_features = []
#         self.features = self.dp.list_columns(print_columns=False)
#         self.features = [f for f in self.features if f not in ['timestamp']]
#     def generate_cross_features(self, drop_features=[]):
#         features = [f for f in self.features if f not in drop_features]
#         new_cols = {}
#         for f1,f2 in combinations(features,2):
#             new_cols[f1 + '_x_' + f2] = self.df[f1] * self.df[f2]
#         self.cross_features = new_cols.keys()
#         self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)
#         return self.cross_features
    
#     def cycle_time_engineer(self):
#         new_cols = {
#         'hour_sin': np.sin(2 * np.pi * self.df['timestamp'].dt.hour / 24),
#         'hour_cos': np.cos(2 * np.pi * self.df['timestamp'].dt.hour / 24),
#         'day_sin':  np.sin(2 * np.pi * self.df['timestamp'].dt.dayofyear / 365),
#         'day_cos':  np.cos(2 * np.pi * self.df['timestamp'].dt.dayofyear / 365),
#         }
#         self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

#     def lag_engineer(self):
#         new_cols = {}
#         for f in self.features:
#             for lag in [1,2,3,6,12]:
#                 new_cols[f + '_lag_' + str(lag)] = self.df[f].shift(lag)
#         self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

#     def rolling_mean_engineer(self):
#         new_cols = {}
#         for f in self.features:
#             for window in [1,2,3,6,12]:
#                 new_cols[f + '_rolling_mean_' + str(window)] = self.df[f].rolling(window).mean()
#         self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

#     def rolling_std_engineer(self):
#         new_cols = {}
#         for f in self.features:
#             for window in [1,2,3,6,12]:
#                 new_cols[f + '_rolling_std_' + str(window)] = self.df[f].rolling(window).std()
#         self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

