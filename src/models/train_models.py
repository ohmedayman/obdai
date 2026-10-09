"""
Smart OBD-II Electrical Diagnostic System
Module: AI Model Training & Evaluation Pipeline

Trains:
1. Isolation Forest (Unsupervised Anomaly Detection for Out-of-Distribution / Electrical Faults)
2. Random Forest Classifier (Multi-Class Diagnostic Classifier: Healthy, Battery, Alternator, Wiring/Drain)
Evaluates metrics (Precision, Recall, F1-Score, Confusion Matrix) and saves artifacts.
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reconfigure stdout for UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.features.feature_engineering import ElectricalFeatureExtractor
from src.generator.synthetic_generator import PATTERN_NAMES, SimulationPattern


def train_and_evaluate_models(
    raw_data_path: Path,
    models_dir: Path,
    test_size: float = 0.25,
    random_state: int = 42,
):
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🤖 SMART OBD-II ELECTRICAL AI MODEL TRAINING PIPELINE ⚡")
    print("=" * 70)
    print(f"📖 Loading raw dataset from: {raw_data_path}")

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}. Run generate_dataset.py first!")

    df = pd.read_csv(raw_data_path)
    print(f"📊 Total Raw Telemetry Records: {len(df):,} rows from {df['session_id'].nunique()} sessions.")

    # 1. Feature Extraction
    print("\n⚙️ Extracting session-level electrical diagnostic features...")
    X, y_multi, y_anomaly = ElectricalFeatureExtractor.transform_dataset_to_feature_matrix(df)
    print(f"✅ Feature Matrix Created: {X.shape[0]} sessions x {X.shape[1]} features.")

    # 2. Split into Train & Test sets
    X_train, X_test, y_train, y_test, y_anom_train, y_anom_test = train_test_split(
        X, y_multi, y_anomaly, test_size=test_size, random_state=random_state, stratify=y_multi
    )
    print(f"🔀 Train Sessions: {len(X_train)} | Test Sessions: {len(X_test)}")

    # 3. Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # MODEL 1: Isolation Forest (Anomaly Detection)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🌲 [1/2] Training Isolation Forest (Anomaly Detector)...")
    
    # Train Isolation Forest on normal sessions
    X_normal_train = X_train_scaled[y_anom_train == 0]
    
    iso_forest = IsolationForest(
        n_estimators=120,
        contamination=0.08,
        random_state=random_state,
    )
    iso_forest.fit(X_normal_train)

    # In sklearn Isolation Forest: 1 = Inlier (Normal), -1 = Outlier (Anomaly)
    raw_preds = iso_forest.predict(X_test_scaled)
    # Map to 0 = Normal, 1 = Anomaly
    iso_preds = np.where(raw_preds == -1, 1, 0)
    
    anom_acc = accuracy_score(y_anom_test, iso_preds)
    anom_f1 = f1_score(y_anom_test, iso_preds, pos_label=1)
    print(f"✅ Isolation Forest Anomaly Detection Accuracy: {anom_acc * 100:.2f}% | F1-Score: {anom_f1:.3f}")

    # -------------------------------------------------------------
    # MODEL 2: Multi-Class Diagnostic Classifier (Random Forest)
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🌳 [2/2] Training Random Forest Diagnostic Classifier...")

    rf_classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=3,
        random_state=random_state,
    )
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    y_proba = rf_classifier.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    print(f"🎯 Classifier Test Accuracy: {acc * 100:.2f}% | Macro F1-Score: {f1_macro:.4f}")

    # Classification Report
    target_names = [f"Class {p.value}: {PATTERN_NAMES[p]['en']}" for p in SimulationPattern]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print("\n📋 Detailed Classification Report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("🔢 Confusion Matrix:")
    print(cm)

    # Feature Importances
    print("\n🌟 Top Feature Importances in Electrical Diagnostics:")
    importances = rf_classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    for rank, idx in enumerate(indices[:8], 1):
        print(f"   {rank}. {X.columns[idx]:<28} : {importances[idx] * 100:.2f}%")

    # -------------------------------------------------------------
    # 4. Save Models & Scalers
    # -------------------------------------------------------------
    iso_path = models_dir / "isolation_forest.joblib"
    rf_path = models_dir / "electrical_classifier.joblib"
    scaler_path = models_dir / "scaler.joblib"
    meta_path = models_dir / "features_meta.joblib"

    joblib.dump(iso_forest, iso_path)
    joblib.dump(rf_classifier, rf_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(ElectricalFeatureExtractor.FEATURE_COLUMNS, meta_path)

    print("\n" + "=" * 70)
    print(f"💾 Models successfully saved to: {models_dir}")
    print(f"   - {iso_path.name}")
    print(f"   - {rf_path.name}")
    print(f"   - {scaler_path.name}")
    print(f"   - {meta_path.name}")
    print("=" * 70)


def main():
    raw_csv = PROJECT_ROOT / "data" / "raw" / "obd_electrical_telemetry_raw.csv"
    models_dir = PROJECT_ROOT / "models"
    train_and_evaluate_models(raw_csv, models_dir)


if __name__ == "__main__":
    main()
