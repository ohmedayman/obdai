"""
Smart OBD-II Electrical Diagnostic System
Module: Feature Engineering & Preprocessing

Extracts aggregated diagnostic electrical features from time-series OBD-II sessions
and windowed telemetry streams for Machine Learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


class ElectricalFeatureExtractor:
    """
    Extracts time-series domain features and instantaneous telemetry representations
    tailored for automotive battery and charging system diagnostics.
    """

    FEATURE_COLUMNS = [
        "resting_voltage_v",
        "min_cranking_voltage_v",
        "cranking_voltage_sag_v",
        "running_voltage_mean_v",
        "running_voltage_std_v",
        "voltage_under_load_mean_v",
        "load_voltage_drop_v",
        "voltage_ripple_mean_v",
        "avg_engine_rpm",
        "avg_coolant_temp_c",
        "max_electrical_load_pct",
    ]

    @classmethod
    def extract_session_features(cls, session_df: pd.DataFrame) -> Dict[str, float]:
        """
        Extracts high-level diagnostic features from a complete or partial trip session.
        """
        # 1. Resting phase (Engine OFF: state = 0)
        resting_subset = session_df[session_df["engine_state"] == 0]
        if not resting_subset.empty:
            resting_voltage = float(resting_subset["battery_voltage_v"].iloc[:15].mean())
        else:
            resting_voltage = float(session_df["battery_voltage_v"].iloc[0])

        # 2. Cranking phase (Engine CRANKING: state = 1)
        cranking_subset = session_df[session_df["engine_state"] == 1]
        if not cranking_subset.empty:
            min_crank_v = float(cranking_subset["battery_voltage_v"].min())
            crank_sag = resting_voltage - min_crank_v
        else:
            min_crank_v = resting_voltage
            crank_sag = 0.0

        # 3. Running phase (Engine IDLE or DRIVING: state >= 2)
        running_subset = session_df[session_df["engine_state"] >= 2]
        if not running_subset.empty:
            running_v_mean = float(running_subset["battery_voltage_v"].mean())
            running_v_std = float(running_subset["battery_voltage_v"].std(ddof=0))
            avg_rpm = float(running_subset["rpm"].mean())
            avg_temp = float(running_subset["coolant_temp_c"].mean())
            ripple_mean = float(running_subset["voltage_ripple_v"].mean())
        else:
            running_v_mean = resting_voltage
            running_v_std = 0.0
            avg_rpm = 0.0
            avg_temp = 25.0
            ripple_mean = 0.02

        # 4. Heavy load phase (Electrical Load > 60%)
        heavy_load_subset = session_df[session_df["electrical_load_pct"] >= 60.0]
        if not heavy_load_subset.empty:
            load_v_mean = float(heavy_load_subset["battery_voltage_v"].mean())
            load_v_drop = max(0.0, running_v_mean - load_v_mean)
        else:
            load_v_mean = running_v_mean
            load_v_drop = 0.0

        max_elec_load = float(session_df["electrical_load_pct"].max()) if not session_df.empty else 0.0

        return {
            "resting_voltage_v": round(resting_voltage, 3),
            "min_cranking_voltage_v": round(min_crank_v, 3),
            "cranking_voltage_sag_v": round(crank_sag, 3),
            "running_voltage_mean_v": round(running_v_mean, 3),
            "running_voltage_std_v": round(running_v_std, 3),
            "voltage_under_load_mean_v": round(load_v_mean, 3),
            "load_voltage_drop_v": round(load_v_drop, 3),
            "voltage_ripple_mean_v": round(ripple_mean, 3),
            "avg_engine_rpm": round(avg_rpm, 1),
            "avg_coolant_temp_c": round(avg_temp, 1),
            "max_electrical_load_pct": round(max_elec_load, 1),
        }

    @classmethod
    def transform_dataset_to_feature_matrix(
        cls, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Transforms raw time-series dataframe into session-level feature matrix.
        Returns: (X, y_multiclass, y_anomaly)
        """
        records = []
        labels = []
        anomaly_labels = []

        grouped = df.groupby("session_id", sort=False)
        for session_id, session_data in grouped:
            features = cls.extract_session_features(session_data)
            features["session_id"] = session_id
            records.append(features)

            labels.append(session_data["label"].iloc[0])
            anomaly_labels.append(session_data["anomaly_label"].iloc[0])

        features_df = pd.DataFrame(records).set_index("session_id")
        y_multiclass = pd.Series(labels, index=features_df.index, name="label")
        y_anomaly = pd.Series(anomaly_labels, index=features_df.index, name="anomaly_label")

        return features_df[cls.FEATURE_COLUMNS], y_multiclass, y_anomaly
