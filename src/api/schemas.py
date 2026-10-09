"""
FastAPI Request and Response Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.rules_nlp.arabic_alert_engine import DiagnosticAlertResponse


class TelemetryInputSchema(BaseModel):
    """Payload representing live OBD-II electrical measurements."""
    session_id: Optional[str] = "SESSION_LIVE_001"
    
    # Core Electrical Measurements
    resting_voltage_v: Optional[float] = Field(12.6, description="Resting battery voltage before cranking (V)")
    min_cranking_voltage_v: Optional[float] = Field(10.5, description="Minimum voltage reached during engine starter crank (V)")
    cranking_voltage_sag_v: Optional[float] = Field(2.1, description="Voltage drop during cranking (V)")
    running_voltage_mean_v: Optional[float] = Field(14.1, description="Average alternator output voltage while running (V)")
    running_voltage_std_v: Optional[float] = Field(0.04, description="Standard deviation of running voltage (V)")
    voltage_under_load_mean_v: Optional[float] = Field(14.0, description="Average voltage when AC/lights are active (V)")
    load_voltage_drop_v: Optional[float] = Field(0.1, description="Voltage drop caused by high electrical accessories (V)")
    voltage_ripple_mean_v: Optional[float] = Field(0.02, description="Short term AC ripple voltage (V)")
    
    # Supporting OBD-II PIDs
    avg_engine_rpm: Optional[float] = Field(800.0, description="Engine RPM")
    avg_coolant_temp_c: Optional[float] = Field(85.0, description="Coolant temperature (C)")
    max_electrical_load_pct: Optional[float] = Field(45.0, description="Maximum accessory load percentage (%)")
    
    # Options
    dtc_codes: Optional[List[str]] = Field(default_factory=list, description="List of active OBD-II Diagnostic Trouble Codes")
    include_llm_advisory: bool = Field(True, description="Whether to include LLM-generated Arabic diagnostic advice")


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class MechanicChatRequest(BaseModel):
    message: str = Field(..., description="Driver's question to the AI Mechanic")
    chat_history: Optional[List[ChatMessage]] = Field(default_factory=list)
    vehicle_context: Optional[Dict[str, Any]] = None


class MechanicChatResponse(BaseModel):
    response_ar: str
    status: str = "success"
    latency_ms: Optional[int] = 0
    model: Optional[str] = "deepseek-chat"
    source: Optional[str] = "LIVE_LLM_API"


class APIConfigSchema(BaseModel):
    api_key: str = Field(..., description="LLM Provider API Key")
    base_url: Optional[str] = Field("https://api.deepseek.com/v1", description="Provider API Base URL")
    model_name: Optional[str] = Field("deepseek-chat", description="Model identifier")


class TestKeyResponseSchema(BaseModel):
    success: bool
    status_code: int
    latency_ms: int
    provider: str
    model: str
    message: Optional[str] = None
    error_message: Optional[str] = None
