"""
Feature Engineering Module
- Cross Features
- Lag Features
- Rolling Mean/Std
- Cycle Time Features
- Physics-Based Features
"""

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

try:
    from data_processor import DataProcessor
except ImportError:
    DataProcessor = None


class FeatureEngineering:
    def __init__(self, dp):
        self.dp = dp
        self.df = dp.df
        self.cross_features = []
        self.features = [f for f in self.df.columns if f not in ['timestamp']]

    def generate_cross_features(self, drop_features=[]):
        features = [f for f in self.features if f not in drop_features]
        new_cols = {}
        for f1, f2 in combinations(features, 2):
            new_cols[f1 + '_x_' + f2] = self.df[f1] * self.df[f2]
        self.cross_features = new_cols.keys()
        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)
        return self.cross_features

    def cycle_time_engineer(self):
        new_cols = {
            'hour_sin': np.sin(2 * np.pi * self.df['timestamp'].dt.hour / 24),
            'hour_cos': np.cos(2 * np.pi * self.df['timestamp'].dt.hour / 24),
            'day_sin': np.sin(2 * np.pi * self.df['timestamp'].dt.dayofyear / 365),
            'day_cos': np.cos(2 * np.pi * self.df['timestamp'].dt.dayofyear / 365),
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
        base_features = [f for f in self.features if f not in drop_features]
        has_cycle = "cycle_id" in self.df.columns

        for f in base_features:
            for lag in lags:
                if mode in ["cycle", "both"] and has_cycle:
                    col_name = f"{f}_cycle_lag_{lag}" if mode == "both" else f"{f}_lag_{lag}"
                    new_cols[col_name] = self.df.groupby("cycle_id")[f].shift(lag)

                if mode in ["global", "both"]:
                    col_name = f"{f}_global_lag_{lag}" if mode == "both" else f"{f}_lag_{lag}"
                    new_cols[col_name] = self.df[f].shift(lag)

        self.df = pd.concat([self.df, pd.DataFrame(new_cols, index=self.df.index)], axis=1)

    def rolling_mean_engineer(self):
        new_cols = {}
        for f in self.features:
            for window in [1, 2, 3, 6, 12]:
                new_cols[f + '_rolling_mean_' + str(window)] = self.df[f].rolling(window).mean()
        self.df = pd.concat([self.df, pd.DataFrame(new_cols)], axis=1)

    def rolling_std_engineer(self):
        new_cols = {}
        for f in self.features:
            for window in [1, 2, 3, 6, 12]:
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
        self.dp.df = self.df

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

        valid_mask = (feed > 0) & feed.notna() & perm.notna()

        frac = np.full(len(self.df), np.nan, dtype=float)
        frac[valid_mask] = 1.0 - (perm[valid_mask] / feed[valid_mask])

        self.df[frac_col] = frac
        self.df[pct_col] = frac * 100.0

        self.dp.df = self.df

        for c in [frac_col, pct_col]:
            if c not in self.features:
                self.features.append(c)

        return frac_col, pct_col


class PhysicsBasedFeatures:
    """
    Physics-based models for RO system parameters including differential pressure,
    permeate pressure, permeate conductivity, and concentrate conductivity.

    Parameters
    ----------
    dp : DataProcessor
        Main data processor holding the full dataset (used for prediction).
    calib_df : pd.DataFrame, optional
        Steady-state data from CycleProcessor.extract_hybrid_cycles() used
        exclusively for calibrating physics parameters.  Should represent
        known-clean, stable operation (ideally the earliest cycles when the
        membrane was freshest).  If None, falls back to the first N rows of
        dp.df with a warning — pass this whenever possible.

    Usage
    -----
        clean_df, summary_df = cp.extract_hybrid_cycles()
        pbf = PhysicsBasedFeatures(dp, calib_df=clean_df)
    """
    def __init__(self, dp, calib_df=None):
        self.dp = dp
        self.df = dp.df
        self.features = [f for f in self.df.columns if f not in ['timestamp']]

        if calib_df is not None:
            # Use only the earliest steady-state data for calibration so that
            # later-cycle fouling does not bias the "clean membrane" baseline.
            self.calib_df = calib_df.sort_values('timestamp').reset_index(drop=True)
            print(f"   PhysicsBasedFeatures: using {len(self.calib_df)} steady-state rows "
                  f"for calibration "
                  f"({self.calib_df['timestamp'].iloc[0]} → "
                  f"{self.calib_df['timestamp'].iloc[-1]})")
        else:
            self.calib_df = None
            print("   ⚠️  PhysicsBasedFeatures: no calib_df supplied — will fall back "
                  "to raw first rows for calibration.  Pass calib_df from "
                  "CycleProcessor.extract_hybrid_cycles() for a stable baseline.")

    def _get_calib_data(self, min_rows, fallback_rows):
        """
        Return the calibration subset to use.

        Prefers self.calib_df (steady-state cycles).  If not available or too
        small, falls back to the first `fallback_rows` rows of the full df.
        `min_rows` is the minimum we need for a reliable fit.
        """
        if self.calib_df is not None and len(self.calib_df) >= min_rows:
            return self.calib_df.copy()
        if self.calib_df is not None:
            print(f"   ⚠️  calib_df only has {len(self.calib_df)} rows "
                  f"(need ≥ {min_rows}). Falling back to first {fallback_rows} "
                  f"rows of full dataset.")
        return self.df.iloc[:fallback_rows].copy()
    
    #-------- DP ------------------------------------------------------------------------------------------------

    def predict_differential_pressure_2(self,df, feed_col='FeedPressure', conc_col='ConcentratePressure'):
        """
        Simple physics-based baseline for Differential Pressure: DP = FeedPressure - ConcentratePressure.
        """
        df['Physics_dP'] = df[feed_col] - df[conc_col]
        return self.df
    
    def create_advanced_dp_model(self, df, flow_col='FeedFlow', temp_col='FeedTemperature', cond_col='FeedConductivity', dp_col='DifferentialPressure'):
        """
        Advanced physics-based baseline for Differential Pressure.
        Incorporates Feed Conductivity to account for salinity-induced viscosity changes.
        """
        
        # 1. Isolate "Clean" Calibration Data (e.g., first 1000 rows)
        calib_df = self._get_calib_data(min_rows=100, fallback_rows=1000).dropna()
        calib_df = calib_df[calib_df[flow_col] > 1]
        
        Q_calib = calib_df[flow_col].values
        T_F_calib = calib_df[temp_col].values
        C_calib = calib_df[cond_col].values
        y_true_dp = calib_df[dp_col].values
        
        # Convert Temp to Celsius
        T_C_calib = (T_F_calib - 32) * 5/9

        # 2. Define the Advanced Physics Equation
        def advanced_physics_equation(X, k, alpha, beta, gamma, c):
            Q_in, T_in, C_in = X
            # Baseline fluid friction * salinity viscosity modifier + offset
            return (k * (Q_in ** beta) * np.exp(alpha * T_in)) * (1 + gamma * C_in) + c

        # 3. Initial Guesses (p0) and bounds
        # Added a very small initial guess for gamma (0.0001) since conductivity values are usually large
        p0 = [1.0, -0.006, 1.8, 0.0001, 0.0]
        bounds = (
            [0.0, -0.05, 1.0, -0.01, -100.0], # Min bounds
            [np.inf, 0.01, 3.0, 0.01, 100.0]  # Max bounds
        )

        print("Calibrating Advanced DP Model (with Conductivity)...")
        try:
            popt, pcov = curve_fit(
                advanced_physics_equation,
                (Q_calib, T_C_calib, C_calib),
                y_true_dp,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )
            best_k, best_alpha, best_beta, best_gamma, best_c = popt
            
            print(f"✅ Optimization Success!")
            print(f" k (Scale):           {best_k:.5f}")
            print(f" α (Temp Coeff):       {best_alpha:.5f}")
            print(f" β (Flow Exp):         {best_beta:.5f}")
            print(f" γ (Cond/Visc Coeff):  {best_gamma:.8f} <-- Professor's Factor!")
            print(f" C (Offset):           {best_c:.5f}")
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return df

        # 4. Predict on the whole dataset
        Q_all = df[flow_col].values
        T_C_all = (df[temp_col].values - 32) * 5/9
        C_all = df[cond_col].values
        
        predicted_dp = advanced_physics_equation((Q_all, T_C_all, C_all), best_k, best_alpha, best_beta, best_gamma, best_c)
        
        df['Advanced_Physics_DP'] = predicted_dp
        # df['Advanced_DP_Residual'] = df[dp_col] - df['Advanced_Physics_DP']

        return df
    
  

    def predict_dp_advanced_physics(self, 
                                    feed_flow_col='FeedFlow', 
                                    temp_col='FeedTemperature', 
                                    cond_col='FeedConductivity', 
                                    target_col='DifferentialPressure'):
        """
        Optimizes the advanced physics model for Differential Pressure:
        dP = k * Q^beta * exp[K * (1/298 - 1/(273+T))] + gamma * C_feed
        """
        print(f"--- Training Advanced TCF-Physics Model for Differential Pressure ---")

        # 1. Prepare Calibration Data
        calib_df = self._get_calib_data(min_rows=100, fallback_rows=1000).dropna(
            subset=[feed_flow_col, temp_col, cond_col, target_col]
        )
        calib_df = calib_df[calib_df[feed_flow_col] > 1] # Filter out plant shutdown periods

        # Extract arrays
        Q = calib_df[feed_flow_col].values
        T_F = calib_df[temp_col].values
        C_feed = calib_df[cond_col].values
        y_true = calib_df[target_col].values

        # Convert Fahrenheit to Celsius
        T_C = (T_F - 32) * 5/9

        # 2. Define your exact mathematical equation
        def physics_equation(X, k, beta, K, gamma):
            Q_in, T_in, C_in = X
            # The official Temperature Correction Factor (TCF) component
            tcf = np.exp(K * ((1.0 / 298.0) - (1.0 / (273.0 + T_in))))
            
            # The full equation
            return (k * (Q_in ** beta) * tcf) + (gamma * C_in)

        # 3. Initial Guesses (p0) and Boundaries (bounds)
        # k: 1.0 (Scale factor)
        # beta: 1.8 (Fluid friction exponent)
        # K: 2640 (Standard membrane activation energy constant)
        # gamma: 0.0001 (Small linear adjustment for conductivity)
        p0 = [1.0, 1.8, 2640.0, 0.0001]
        
        bounds = (
            [0.0, 1.0, 1000.0, -0.05],  # Lower bounds
            [np.inf, 3.0, 4000.0, 0.05] # Upper bounds
        )

        try:
            # 4. Train the model (optimize parameters)
            popt, pcov = curve_fit(
                physics_equation,
                (Q, T_C, C_feed),
                y_true,
                p0=p0,
                bounds=bounds,
                maxfev=15000 # Increased max evaluations because K is a large number
            )
            best_k, best_beta, best_K, best_gamma = popt

            print(f"✅ Optimization Success!")
            print(f"   k (Scale):       {best_k:.5f}")
            print(f"   β (Flow Exp):    {best_beta:.5f}")
            print(f"   K (TCF Const):   {best_K:.1f}")
            print(f"   γ (Cond Coeff):  {best_gamma:.7f}")

        except Exception as e:
            print(f"⚠️ Optimization failed: {e}")
            # Fallback to standard theoretical values if the solver fails
            best_k, best_beta, best_K, best_gamma = 1.0, 1.8, 2640.0, 0.0001

        # 5. Apply the trained equation to the entire dataset
        Q_all = self.df[feed_flow_col].values
        T_F_all = self.df[temp_col].values
        C_feed_all = self.df[cond_col].values
        T_C_all = (T_F_all - 32) * 5/9

        # Calculate final predictions
        tcf_all = np.exp(best_K * ((1.0 / 298.0) - (1.0 / (273.0 + T_C_all))))
        dP_pred = (best_k * (Q_all ** best_beta) * tcf_all) + (best_gamma * C_feed_all)

        # Save to dataframe
        self.df['Physics_DifferentialPressure'] = dP_pred

        return self.df
    

    def predict_differential_pressure_physics_v4(self, 
                                                feed_flow_col='FeedFlow', 
                                                temp_col='FeedTemperature', 
                                                cond_col='FeedConductivity', 
                                                target_col='DifferentialPressure'):
        """
        Optimizes k, alpha, beta, gamma, and C simultaneously.
        Incorporates Feed Flow, Temperature, and Conductivity.
        """
        print(f"--- Generated Physics Model for DifferentialPressure (incl. Conductivity) ---")

        calib_df = self._get_calib_data(min_rows=100, fallback_rows=1000).dropna(
            subset=[feed_flow_col, temp_col, cond_col, target_col]
        )
        calib_df = calib_df[calib_df[feed_flow_col] > 1]

        # 1. Extract the three input arrays
        Q = calib_df[feed_flow_col].values
        T_F = calib_df[temp_col].values
        Cond = calib_df[cond_col].values
        y_true = calib_df[target_col].values

        T_C = (T_F - 32) * 5/9

        # 2. Update the physics equation to accept a tuple of 3 inputs
        def physics_equation(X, k, alpha, beta, gamma, c):
            Q_in, T_in, Cond_in = X
            return (k * (Q_in ** beta) * np.exp(alpha * T_in + gamma * Cond_in)) + c

        # 3. Add initial guess (p0) and bounds for the new gamma parameter
        # Initial guess for gamma is very small (e.g., 0.0001) because conductivity values are often large (e.g., 1000+ µS/cm)
        p0 = [1.0, -0.006, 1.8, 0.0001, 0.0]
        bounds = (
            [0.0, -0.05, 1.0, -0.01, -100.0],  # Lower bounds (added gamma)
            [np.inf, 0.01, 3.0, 0.01, 100.0]   # Upper bounds (added gamma)
        )

        try:
            # Pass the 3 variables as a tuple to the optimizer
            popt, pcov = curve_fit(
                physics_equation,
                (Q, T_C, Cond),
                y_true,
                p0=p0,
                bounds=bounds,
                maxfev=10000
            )
            best_k, best_alpha, best_beta, best_gamma, best_c = popt

            print(f"✅ Optimization Success!")
            print(f"   k (Scale):       {best_k:.5f}")
            print(f"   α (Temp Coeff):  {best_alpha:.5f}")
            print(f"   β (Flow Exp):    {best_beta:.5f}")
            print(f"   γ (Cond Coeff):  {best_gamma:.7f}")
            print(f"   C (Offset):      {best_c:.5f}")

        except Exception as e:
            print(f"⚠️ Optimization failed: {e}")
            # Fallback values if it fails
            best_k, best_alpha, best_beta, best_gamma, best_c = 1.0, -0.006, 1.7, 0.0001, 0.0

        # 4. Apply the optimized equation to the whole dataset
        Q_all = self.df[feed_flow_col].values
        T_F_all = self.df[temp_col].values
        Cond_all = self.df[cond_col].values
        T_C_all = (T_F_all - 32) * 5/9

        dP_pred = (best_k * (Q_all ** best_beta) * np.exp(best_alpha * T_C_all + best_gamma * Cond_all)) + best_c

        self.df['Physics_DifferentialPressure_V4'] = dP_pred

        return self.df
    
    def predict_differential_pressure_physics_V3(
        self,
        feed_flow_col='FeedFlow',
        temp_col='FeedTemperature',
        cond_col='FeedConductivity',
        target_col='DifferentialPressure'
    ):
        """
        Physics-based prediction of differential pressure across the RO membrane.

        Model:
            ΔP = k · Q^β · exp(α · T_C) + γ · C_feed + c

        Where:
            Q       : Feed flow rate
            T_C     : Feed temperature in °C (converted from °F if needed)
            C_feed  : Feed conductivity (proxy for osmotic pressure / TDS)
            k       : Hydraulic resistance coefficient
            α       : Temperature coefficient (Arrhenius-type viscosity correction)
            β       : Flow exponent (1.0=laminar, 2.0=turbulent; spacer-filled RO ≈ 1.4–1.7)
            γ       : Conductivity scaling factor (osmotic pressure contribution)
            c       : Offset term

        References:
            - Schock & Miquel (1987), Desalination 64, 339–352  [hydraulic friction / β]
            - Sharqawy et al. (2010), Desal. Wat. Treat. 16, 354–380  [Arrhenius viscosity]
            - Wijmans & Baker (1995), J. Membr. Sci. 107, 1–21  [solution-diffusion / conductivity]

        Calibration uses steady-state cycle data (calib_df) when available,
        otherwise falls back to the first 1000 rows of the full dataset.
        """
        print(f"--- Generated Physics Model for {target_col} ---")

        # ── Calibration data ────────────────────────────────────────────────────
        calib_df = self._get_calib_data(min_rows=100, fallback_rows=1000).dropna(
            subset=[feed_flow_col, temp_col, cond_col, target_col]
        )
        calib_df = calib_df[calib_df[feed_flow_col] > 1]

        Q      = calib_df[feed_flow_col].values
        T_F    = calib_df[temp_col].values
        C      = calib_df[cond_col].values
        y_true = calib_df[target_col].values

        # Convert °F → °C
        T_C = (T_F - 32) * 5 / 9

        # ── Physics equation ────────────────────────────────────────────────────
        def physics_equation(X, k, alpha, beta, gamma, c):
            Q_in, T_in, C_in = X
            hydraulic = k * (Q_in ** beta) * np.exp(alpha * T_in)
            osmotic   = gamma * C_in
            return hydraulic + osmotic + c

        p0 = [1.0, -0.006, 1.8, 0.0001, 0.0]
        bounds = (
            [0.0,    -0.05, 1.0, 0.0,    -100.0],   # lower
            [np.inf,  0.01, 3.0, np.inf,  100.0]    # upper
        )

        # ── Fit ─────────────────────────────────────────────────────────────────
        try:
            popt, pcov = curve_fit(
                physics_equation,
                (Q, T_C, C),
                y_true,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )
            best_k, best_alpha, best_beta, best_gamma, best_c = popt

            print(f"✅ Optimization Success!")
            print(f"   k (Scale):          {best_k:.5f}")
            print(f"   α (Temp Coeff):     {best_alpha:.5f}")
            print(f"   β (Flow Exp):       {best_beta:.5f}")
            print(f"   γ (Cond Factor):    {best_gamma:.5f}")
            print(f"   C (Offset):         {best_c:.5f}")

        except Exception as e:
            print(f"⚠️ Optimization failed: {e}")
            print(f"   Using default fallback parameters.")
            best_k, best_alpha, best_beta, best_gamma, best_c = 1.0, -0.006, 1.7, 0.0001, 0.0

        # ── Predict on full dataset ─────────────────────────────────────────────
        Q_all   = self.df[feed_flow_col].values
        T_F_all = self.df[temp_col].values
        C_all   = self.df[cond_col].values
        T_C_all = (T_F_all - 32) * 5 / 9

        dP_pred = (
            best_k * (Q_all ** best_beta) * np.exp(best_alpha * T_C_all)
            + best_gamma * C_all
            + best_c
        )

        self.df['Physics_DifferentialPressure_V3'] = dP_pred

        return self.df



    def predict_differential_pressure_physics(self, feed_flow_col='FeedFlow', temp_col='FeedTemperature', target_col='DifferentialPressure'):
        """
        Optimizes k, alpha, beta, and C (offset) simultaneously.
        Calibration uses steady-state cycle data (calib_df) when available,
        otherwise falls back to the first 1000 rows of the full dataset.
        """
        print(f"--- Generated Physics Model for DifferentialPressure ---")

        calib_df = self._get_calib_data(min_rows=100, fallback_rows=1000).dropna()
        calib_df = calib_df[calib_df[feed_flow_col] > 1]

        Q = calib_df[feed_flow_col].values
        T_F = calib_df[temp_col].values
        y_true = calib_df[target_col].values

        T_C = (T_F - 32) * 5/9

        def physics_equation(X, k, alpha, beta, c):
            Q_in, T_in = X
            return (k * (Q_in ** beta) * np.exp(alpha * T_in)) + c

        p0 = [1.0, -0.006, 1.8, 0.0]
        bounds = (
            [0.0, -0.05, 1.0, -100.0],
            [np.inf, 0.01, 3.0, 100.0]
        )

        try:
            popt, pcov = curve_fit(
                physics_equation,
                (Q, T_C),
                y_true,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )
            best_k, best_alpha, best_beta, best_c = popt

            print(f"✅ Optimization Success!")
            print(f"   k (Scale):     {best_k:.5f}")
            print(f"   α (Temp Coeff): {best_alpha:.5f}")
            print(f"   β (Flow Exp):   {best_beta:.5f}")
            print(f"   C (Offset):     {best_c:.5f}")

        except Exception as e:
            print(f"⚠️ Optimization failed: {e}")
            best_k, best_alpha, best_beta, best_c = 1.0, -0.006, 1.7, 0.0

        Q_all = self.df[feed_flow_col].values
        T_F_all = self.df[temp_col].values
        T_C_all = (T_F_all - 32) * 5/9

        dP_pred = (best_k * (Q_all ** best_beta) * np.exp(best_alpha * T_C_all)) + best_c

        self.df['Physics_DifferentialPressure'] = dP_pred

        # return dP_pred, (best_k, best_alpha, best_beta, best_c)
        return self.df

#----------------------------------- Permeate Conductivity ------------------------------------------------
    def predict_permeate_conductivity_physics(self,
                                              perm_flow_col='PermeateFlow',
                                              feed_cond_col='FeedConductivity',
                                              temp_col='FeedTemperature',
                                              calibration_col='PermeateConductivity',
                                              rolling_window_size=168):
        """
        Calculates Theoretical Permeate Conductivity using Dynamic Salt Passage.
        Adapts to membrane fouling over time using a rolling B coefficient.
        """
        print("\n--- Generated Dynamic Physics Model for Permeate Conductivity ---")

        Q_perm = self.df[perm_flow_col].values
        C_feed = self.df[feed_cond_col].values
        T = self.df[temp_col].values
        C_perm_actual = self.df[calibration_col].values

        if 'Physics_ConcentrateConductivity' in self.df.columns:
            C_conc = self.df['Physics_ConcentrateConductivity'].values
        else:
            if 'Recovery' in self.df.columns:
                Rec = self.df['Recovery'].values / 100.0
                C_conc = C_feed * (1 / (1 - Rec))
            else:
                C_conc = C_feed * 4.0

        with np.errstate(divide='ignore', invalid='ignore'):
            Driving_Conc = (C_conc - C_feed) / np.log(C_conc / C_feed)
            Driving_Conc = np.nan_to_num(Driving_Conc, nan=np.mean(C_feed))

        T_c = (T - 32) * 5/9
        c = 2640 * ((1/298) - (1/(273 + T_c)))
        TCF = np.exp(c)

        Salt_Flux_Actual = Q_perm * C_perm_actual
        Driving_Potential = Driving_Conc * TCF
        Driving_Potential_Safe = np.where(Driving_Potential < 0.1, 0.1, Driving_Potential)

        B_series = Salt_Flux_Actual / Driving_Potential_Safe

        valid_mask = (Q_perm > 1.0) & (C_perm_actual > 0.1) & (B_series < np.percentile(B_series, 99))

        B_clean = pd.Series(np.where(valid_mask, B_series, np.nan))
        B_filled = B_clean.ffill().bfill()
        B_dynamic = B_filled.rolling(window=rolling_window_size, min_periods=1).median().values

        print(f"Global Median B: {np.nanmedian(B_dynamic):.5f}")
        print(f"Current B (End of dataset): {B_dynamic[-1]:.5f}")

        Q_perm_safe = np.where(Q_perm <= 1.0, 1.0, Q_perm)

        Salt_Flux_Pred = B_dynamic * Driving_Conc * TCF
        C_phys = Salt_Flux_Pred / Q_perm_safe

        self.df['Physics_PermeateConductivity'] = C_phys
        # self.df['Physics_B_dynamic'] = B_dynamic

        return self.df

    def predict_concentrate_conductivity_physics(self, feed_flow_col='FeedFlow', feed_cond_col='FeedConductivity',
                                                  perm_flow_col='PermeateFlow', perm_cond_col='PermeateConductivity',
                                                  conc_flow_col='ConcentrateFlow'):
        """
        Using Mass Balance to obtain physics-based concentration conductivity
        """

        Q_feed = self.df[feed_flow_col].values
        C_feed = self.df[feed_cond_col].values
        Q_perm = self.df[perm_flow_col].values
        C_perm = self.df[perm_cond_col].values
        Q_conc = self.df[conc_flow_col].values

        M_feed = Q_feed * C_feed
        M_perm = Q_perm * C_perm

        Q_conc_safe = np.where(Q_conc <= 0.1, 0.1, Q_conc)

        C_conc = (M_feed - M_perm) / Q_conc_safe

        self.df['Physics_ConcentrateConductivity'] = C_conc

        return self.df

    def predict_concentrate_flow_physics(self, feed_flow_col='FeedFlow', perm_flow_col='PermeateFlow'):
        """
        Using Mass Balance to obtain physics-based concentrate flow
        """

        Q_feed = self.df[feed_flow_col].values
        Q_perm = self.df[perm_flow_col].values

        Q_conc = Q_feed - Q_perm

        self.df['Physics_ConcentrateFlow'] = Q_conc

        print(f"\n --- Generated Physics-Based Concentrate Flow --- ")

        return self.df
    

#--------------------- CP --------------------------------------------------------------- 
    def predict_concentrate_pressure_physics(self, feed_pressure_col='FeedPressure', perm_pressure_col='PermeatePressure'):
        """
        Using Pressure Balance to obtain physics-based concentrate pressure
        """

        P_feed = self.df[feed_pressure_col].values
        P_perm = self.df[perm_pressure_col].values

        P_conc = P_feed - P_perm

        self.df['Physics_ConcentratePressure'] = P_conc

        print(f"\n --- Generated Physics-Based Concentrate Pressure --- ")

        return self.df

    def prediction_concentrate_pressure_physics_V2(self, feed_pressure_col='FeedPressure', diff_pressure_col='DifferentialPressure'):
        '''
        CP = FP - DP
        '''
        P_conc = self.df[feed_pressure_col].values - self.df[diff_pressure_col].values
        self.df['Physics_ConcentratePressure_V2'] = P_conc
        
        print(f'\n --- Generated Physics-Based Concentrate Pressure V2 --- ')
        
        return self.df



    def predict_all_physics(self):
        self.predict_differential_pressure_physics()
        self.predict_permeate_conductivity_physics()
        # self.predict_concentrate_conductivity_physics()
        self.predict_concentrate_flow_physics()
        self.predict_concentrate_pressure_physics()
        return self.df
    