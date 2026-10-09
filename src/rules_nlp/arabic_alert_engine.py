"""
Smart OBD-II Electrical Diagnostic System
Module: Arabic Rule/NLP Alert Engine (محرك التنبيهات والترجمة باللغة العربية)

Converts raw ML predictions, anomaly flags, and telemetry metrics into structured,
human-readable Arabic alerts, risk assessments, and step-by-step actionable recommendations.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DiagnosticAlertResponse(BaseModel):
    """Structured response for mobile application integration."""
    session_id: Optional[str] = None
    status_code: str = Field(..., description="Machine readable status code")
    predicted_class: int = Field(..., description="Predicted class index (0-3)")
    predicted_class_name_en: str
    predicted_class_name_ar: str
    risk_level: str = Field(..., description="Risk severity: LOW, MEDIUM, CRITICAL")
    risk_level_ar: str
    is_anomaly: bool = Field(..., description="Anomaly flag from Isolation Forest")
    safe_to_drive: bool = Field(..., description="Whether it is safe to keep driving")
    confidence_pct: float = Field(..., description="Model confidence percentage")
    
    # Arabic Driver-Facing Content
    title_ar: str = Field(..., description="Brief alert title in Arabic")
    summary_ar: str = Field(..., description="Clear explanation for the driver")
    recommended_action_ar: str = Field(..., description="Actionable recommendation in Arabic")
    technical_details_ar: List[str] = Field(default_factory=list)
    
    # Telemetry Snapshot
    telemetry_summary: Dict[str, Any] = Field(default_factory=dict)
    
    # Optional LLM Advisory
    llm_advisory_ar: Optional[str] = None


class ArabicAlertEngine:
    """
    Translates electrical diagnostic predictions into rich Arabic alerts and advice.
    """

    @classmethod
    def generate_alert(
        cls,
        predicted_class: int,
        confidence_pct: float,
        is_anomaly: bool,
        telemetry: Dict[str, Any],
        llm_advisory_text: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> DiagnosticAlertResponse:
        """
        Generates structured Arabic alert object based on classified fault pattern.
        """
        resting_v = telemetry.get("resting_voltage_v", 12.6)
        crank_v = telemetry.get("min_cranking_voltage_v", 10.5)
        running_v = telemetry.get("running_voltage_mean_v", 14.1)
        load_drop = telemetry.get("load_voltage_drop_v", 0.05)
        ripple_v = telemetry.get("voltage_ripple_mean_v", 0.02)

        tech_details = [
            f"جهد البطارية أثناء السكون: {resting_v:.2f} فولت",
            f"أدنى جهد أثناء بدء التشغيل: {crank_v:.2f} فولت",
            f"متوسط شحن الدينامو أثناء الدوران: {running_v:.2f} فولت",
            f"هبوط الجهد تحت الحمل الكهربائي: {load_drop:.2f} فولت",
            f"مستوى التموج والتذبذب: {ripple_v:.3f} فولت",
        ]

        if predicted_class == 0:
            return DiagnosticAlertResponse(
                session_id=session_id,
                status_code="SYS_HEALTHY",
                predicted_class=0,
                predicted_class_name_en="Healthy Normal State",
                predicted_class_name_ar="المنظومة الكهربائية سليمة ومستقرة",
                risk_level="LOW",
                risk_level_ar="منخفض (آمن تماماً)",
                is_anomaly=is_anomaly,
                safe_to_drive=True,
                confidence_pct=round(confidence_pct, 1),
                title_ar="المنظومة الكهربائية تعمل بكفاءة ممتازة ✅",
                summary_ar="جهد البطارية والشحن الكهربائي من الدينامو ضمن الحدود الهندسية القياسية، ولا توجد أي مؤشرات لتسريب تيار أو هبوط في الجهد.",
                recommended_action_ar="لا يتطلب أي إجراء صيانة حالياً. استمر في القيادة بشكل طبيعي مع الفحص الدوري المعتاد.",
                technical_details_ar=tech_details,
                telemetry_summary=telemetry,
                llm_advisory_ar=llm_advisory_text,
            )

        elif predicted_class == 1:
            return DiagnosticAlertResponse(
                session_id=session_id,
                status_code="WARN_BATTERY_DEGRADATION",
                predicted_class=1,
                predicted_class_name_en="Battery Degradation / Weak Cranking",
                predicted_class_name_ar="تدهور البطارية / ضعف قدرة التدوير",
                risk_level="MEDIUM" if crank_v > 8.5 else "CRITICAL",
                risk_level_ar="متوسط إلى حرج" if crank_v <= 8.5 else "متوسط",
                is_anomaly=is_anomaly,
                safe_to_drive=True,
                confidence_pct=round(confidence_pct, 1),
                title_ar="تحذير: هبوط حاد في جهد البطارية عند بدء التشغيل ⚠️",
                summary_ar=f"تم رصد هبوط في الجهد إلى ({crank_v:.1f} فولت) أثناء محاولة تشغيل السيارة، وهو أقل من الحد الآمن (9.6 فولت). هذا يشير لتآكل خلايا البطارية أو اقتراب انتهاء عمرها الافتراضي.",
                recommended_action_ar="1. تجنب إطفاء المحرك في أماكن بعيدة عن مراكز الصيانة.\n2. افحص أصابع وكابلات البطارية وتأكد من خلوها من الأملاح.\n3. توجه لأقرب مركز صيانة لإجراء اختبار كفاءة تيار التدوير (CCA Test) واستبدال البطارية إن لزم.",
                technical_details_ar=tech_details,
                telemetry_summary=telemetry,
                llm_advisory_ar=llm_advisory_text,
            )

        elif predicted_class == 2:
            is_overcharge = running_v > 14.8
            status = "ERR_ALTERNATOR_OVERCHARGE" if is_overcharge else "ERR_ALTERNATOR_UNDERCHARGE"
            title = "خطر: شحن زائد من الدينامو (Overvoltage) 🚨" if is_overcharge else "خطر: ضعف أو انقطاع شحن الدينامو 🚨"
            summary = (
                f"منظم الجهد بالدينامو يزود الشبكة بجهد خطير ({running_v:.1f} فولت)، مما يهدد بإتلاف الحواسيب الإلكترونية للسيارة (ECUs) وغليان سائل البطارية."
                if is_overcharge
                else f"الدينامو لا يقوم بتعويض الطاقة المستهلكة (الجهد: {running_v:.1f} فولت)، والسيارة تعمل كلياً على طاقة البطارية مما سيؤدي لانطفاء المحرك قريباً."
            )
            return DiagnosticAlertResponse(
                session_id=session_id,
                status_code=status,
                predicted_class=2,
                predicted_class_name_en="Alternator System Fault",
                predicted_class_name_ar="خلل في منظومة شحن الدينامو",
                risk_level="CRITICAL",
                risk_level_ar="حرج جداً (خطر انطفاء أو تلف)",
                is_anomaly=is_anomaly,
                safe_to_drive=False,
                confidence_pct=round(confidence_pct, 1),
                title_ar=title,
                summary_ar=summary,
                recommended_action_ar="1. أطفئ المكيف والأنوار الإضافية وأي أجهزة كهربائية غير ضرورية فوراً لتوفير الطاقة.\n2. توجه فوراً لأقرب فني لفحص سير الدينامو، الفحمات، ومنظم الجهد الداخلي.",
                technical_details_ar=tech_details,
                telemetry_summary=telemetry,
                llm_advisory_ar=llm_advisory_text,
            )

        else:  # predicted_class == 3
            return DiagnosticAlertResponse(
                session_id=session_id,
                status_code="WARN_PARASITIC_DRAIN_OR_RESISTANCE",
                predicted_class=3,
                predicted_class_name_en="Parasitic Drain / High Resistance",
                predicted_class_name_ar="تسريب كهربائي / مقاومة تأريض وأسلاك مرتفعة",
                risk_level="MEDIUM",
                risk_level_ar="متوسط",
                is_anomaly=is_anomaly,
                safe_to_drive=True,
                confidence_pct=round(confidence_pct, 1),
                title_ar="تنبيه: رصد هبوط جهد غير مبرر أو تسريب تيار ⚡",
                summary_ar=f"يحدث هبوط ملحوظ في الجهد بمقدار ({load_drop:.2f} فولت) بمجرد تشغيل المكيف أو الأحمال، أو يتم استنزاف البطارية بمعدل غير طبيعي أثناء توقف السيارة.",
                recommended_action_ar="1. افحص كابل التأريض الرئيسي (Ground Wire) وتأكد من إحكام ربطه بهيكل السيارة.\n2. تأكد من عدم وجود ملحقات إضافية (مثل شاشات، كاميرات، أجهزة إنذار) تسحب تياراً بعد إطفاء المفتاح.",
                technical_details_ar=tech_details,
                telemetry_summary=telemetry,
                llm_advisory_ar=llm_advisory_text,
            )
