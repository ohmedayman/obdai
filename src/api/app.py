"""
Smart OBD-II Electrical Diagnostic System
Module: FastAPI Inference & Advisory Server

Provides high-performance REST APIs for Mobile Application (Flutter / Android / iOS)
and ESP32 edge telemetry gateways.
"""

import os
import sys
from pathlib import Path
from typing import Dict
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.api.schemas import (
    DiagnosticAlertResponse,
    MechanicChatRequest,
    MechanicChatResponse,
    TelemetryInputSchema,
)
from src.rules_nlp.arabic_alert_engine import ArabicAlertEngine
from src.rules_nlp.llm_advisory import AutomotiveLLMAdvisor

app = FastAPI(
    title="Smart OBD-II Electrical Diagnostic AI API",
    description="Predictive maintenance and Arabic AI mechanic advisor for vehicle electrical systems.",
    version="1.0.0",
)

# Enable CORS for Mobile Apps (Flutter, React Native, Web Dashboards)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Trained Models into Memory
MODELS_DIR = PROJECT_ROOT / "models"
try:
    iso_model = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    rf_model = joblib.load(MODELS_DIR / "electrical_classifier.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_cols = joblib.load(MODELS_DIR / "features_meta.joblib")
    print("✅ AI Models successfully loaded into FastAPI memory.")
except Exception as e:
    print(f"⚠️ Warning: Could not pre-load models: {e}. Run train_models.py!")
    iso_model, rf_model, scaler, feature_cols = None, None, None, []

# Initialize LLM Advisor
advisor = AutomotiveLLMAdvisor()


from fastapi.responses import HTMLResponse

STATIC_INDEX_PATH = PROJECT_ROOT / "src" / "api" / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
def root():
    """Serves the interactive Arabic AI Mechanic & Telemetry Web Dashboard."""
    if STATIC_INDEX_PATH.exists():
        with open(STATIC_INDEX_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Smart OBD-II Electrical AI Backend Online</h2><p><a href='/docs'>Swagger API Docs</a></p>")


@app.get("/api/status")
def system_status():
    return {
        "system": "Smart OBD-II Electrical Diagnostic System",
        "version": "1.0.0",
        "status": "Online",
        "models_loaded": rf_model is not None,
        "docs_url": "/docs",
    }



@app.post("/api/predict", response_model=DiagnosticAlertResponse)
def predict_electrical_health(payload: TelemetryInputSchema):
    """
    Evaluates live or aggregated OBD-II electrical telemetry, classifies health status,
    detects anomalies, and generates actionable Arabic alert and AI mechanic advice.
    """
    if rf_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Please train models first.")

    # Convert payload into feature dictionary
    telemetry_dict = {
        "resting_voltage_v": payload.resting_voltage_v,
        "min_cranking_voltage_v": payload.min_cranking_voltage_v,
        "cranking_voltage_sag_v": payload.cranking_voltage_sag_v,
        "running_voltage_mean_v": payload.running_voltage_mean_v,
        "running_voltage_std_v": payload.running_voltage_std_v,
        "voltage_under_load_mean_v": payload.voltage_under_load_mean_v,
        "load_voltage_drop_v": payload.load_voltage_drop_v,
        "voltage_ripple_mean_v": payload.voltage_ripple_mean_v,
        "avg_engine_rpm": payload.avg_engine_rpm,
        "avg_coolant_temp_c": payload.avg_coolant_temp_c,
        "max_electrical_load_pct": payload.max_electrical_load_pct,
    }

    # Build feature row in exact order
    feat_df = pd.DataFrame([telemetry_dict])[feature_cols]

    # 1. Anomaly Detection via Isolation Forest
    feat_scaled = scaler.transform(feat_df)
    iso_raw = iso_model.predict(feat_scaled)[0]
    is_anomaly = bool(iso_raw == -1)

    # 2. Multi-Class Classification via Random Forest
    pred_class = int(rf_model.predict(feat_df)[0])
    probabilities = rf_model.predict_proba(feat_df)[0]
    confidence_pct = float(probabilities[pred_class] * 100.0)

    # 3. Optional LLM Advisory Integration
    llm_advisory_text = None
    if payload.include_llm_advisory:
        ml_prediction_info = {
            "label": pred_class,
            "predicted_class_ar": ArabicAlertEngine.generate_alert(
                predicted_class=pred_class,
                confidence_pct=confidence_pct,
                is_anomaly=is_anomaly,
                telemetry=telemetry_dict,
            ).predicted_class_name_ar,
            "confidence_pct": confidence_pct,
        }
        advisory_result = advisor.generate_advisory_report(
            telemetry_summary=telemetry_dict,
            ml_prediction=ml_prediction_info,
            dtc_codes=payload.dtc_codes,
        )
        llm_advisory_text = advisory_result.get("advisory_text_ar")

    # 4. Generate Structured Arabic Alert
    alert = ArabicAlertEngine.generate_alert(
        predicted_class=pred_class,
        confidence_pct=confidence_pct,
        is_anomaly=is_anomaly,
        telemetry=telemetry_dict,
        llm_advisory_text=llm_advisory_text,
        session_id=payload.session_id,
    )

    return alert


@app.post("/api/advisor/chat", response_model=MechanicChatResponse)
def chat_with_mechanic_endpoint(request: MechanicChatRequest):
    """
    Interactive conversation with AI Mechanic Consultant.
    """
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
    answer = advisor.chat_with_mechanic(
        chat_history=history_dicts,
        user_message=request.message,
        vehicle_context=request.vehicle_context,
    )
    return MechanicChatResponse(response_ar=answer)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
