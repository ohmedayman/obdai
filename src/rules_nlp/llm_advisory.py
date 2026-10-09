"""
Smart OBD-II Electrical Diagnostic System
Module: Advanced LLM Diagnostic & Advisory Engine with Live Provider Validation

Supports OpenAI-compatible endpoints (DeepSeek, OpenAI, Groq, Together, Ollama, OpenRouter)
Features:
- Real-time connection testing & latency measurement
- Live status reporting & error diagnostics (e.g. 401 Invalid Key, Quota Exceeded, Timeout)
- Dynamic runtime reconfiguration from Web UI
- Context-rich automotive prompt engineering for deep diagnostic reasoning
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv, set_key

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class AutomotiveLLMAdvisor:
    """
    Direct LLM Diagnostic Consultant with real-time API connectivity and error reporting.
    """

    SYSTEM_PROMPT = """أنت "خبير واستشاري تشخيص كهرباء السيارات الذكي" (Master OBD-II Automotive Electrical AI Engineer).
تتلقى قراءات الحساسات الكهربائية الحية للسيارة (جهد البطارية، شحن الدينامو، هبوط الجهد عند التشغيل، تذبذب التيار AC Ripple، الـ RPM، والأحمال الكهربائية) ومخرجات نماذج تعلم الآلة.

مهمتك:
1. تحليل القراءات الهندسية بعمق وربطها بالفيزياء الكهربائية للسيارة (مقاومة داخلية، ملفات الدينامو، أقطاب البطارية، دوائر التأريض).
2. تقديم تشخيص دقيق باللغة العربية الواضحة والشاملة لسائق السيارة والميكانيكي.
3. الإجابة بدقة عن:
   - ما هو السبب الجذري للمشكلة؟
   - ما هو مستوى الخطورة الفعلي (منخفض / متوسط / حرج جداً)؟
   - هل القيادة آمنة حالياً أم يجب التوقف الفوري لتجنب احتراق الـ ECUs أو انطفاء المحرك؟
   - خطوات الصيانة والإصلاح المقترحة خطوة بخطوة (فحص أصابع البطارية، شد السير، قياس الأمبير، إلخ).
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_sec: int = 20,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "sk-8e872c0d18aed5b33cf2adbe5cdbbbeccfe17c4e131436bf9459a0899ff8c3f6")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.5abeer.ai/v1")).rstrip("/")
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-5.2")
        self.timeout_sec = timeout_sec


    def update_config(self, api_key: str, base_url: Optional[str] = None, model_name: Optional[str] = None) -> None:
        """Dynamically updates the LLM configuration at runtime and saves to .env."""
        if api_key:
            self.api_key = api_key.strip()
            set_key(str(ENV_PATH), "LLM_API_KEY", self.api_key)
        if base_url:
            self.base_url = base_url.strip().rstrip("/")
            set_key(str(ENV_PATH), "LLM_BASE_URL", self.base_url)
        if model_name:
            self.model_name = model_name.strip()
            set_key(str(ENV_PATH), "LLM_MODEL", self.model_name)

    def test_connection(self) -> Dict[str, Any]:
        """
        Pings the LLM API provider and measures latency, verifying key validity.
        """
        if not self.api_key or len(self.api_key) < 5:
            return {
                "success": False,
                "status_code": 400,
                "latency_ms": 0,
                "provider": self.base_url,
                "model": self.model_name,
                "error_message": "مفتاح الـ API فارغ أو غير معين. يرجى إدخال مفتاح صالح في الإعدادات.",
            }

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        test_payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 5,
        }

        start_time = time.time()
        try:
            resp = requests.post(endpoint, headers=headers, json=test_payload, timeout=self.timeout_sec)
            latency = int((time.time() - start_time) * 1000)

            if resp.status_code == 200:
                return {
                    "success": True,
                    "status_code": 200,
                    "latency_ms": latency,
                    "provider": self.base_url,
                    "model": self.model_name,
                    "message": f"تم الاتصال بنجاح بمزود الذكاء الاصطناعي ({self.model_name}) في {latency}ms!",
                }
            else:
                err_data = {}
                try:
                    err_data = resp.json()
                except Exception:
                    pass
                err_msg = err_data.get("error", {}).get("message", resp.text[:200]) if isinstance(err_data.get("error"), dict) else resp.text[:200]
                
                return {
                    "success": False,
                    "status_code": resp.status_code,
                    "latency_ms": latency,
                    "provider": self.base_url,
                    "model": self.model_name,
                    "error_message": f"فشل التحقق من المزود (كود {resp.status_code}): {err_msg}",
                }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "status_code": 408,
                "latency_ms": int((time.time() - start_time) * 1000),
                "provider": self.base_url,
                "model": self.model_name,
                "error_message": "انتهت مهلة الاتصال بالخادم (Timeout). تحقق من اتصال الإنترنت.",
            }
        except Exception as e:
            return {
                "success": False,
                "status_code": 500,
                "latency_ms": 0,
                "provider": self.base_url,
                "model": self.model_name,
                "error_message": f"خطأ في الاتصال: {str(e)}",
            }

    def generate_advisory_report(
        self,
        telemetry_summary: Dict[str, Any],
        ml_prediction: Dict[str, Any],
        dtc_codes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generates rich diagnostic report directly using the live LLM API.
        """
        dtc_text = ", ".join(dtc_codes) if dtc_codes else "لا توجد أكواد أعطال مسجلة (No DTCs)"
        
        prompt = f"""
تحليل القياسات الكهربائية اللحظية للسيارة (Live OBD-II Telemetry):
- جهد البطارية قبل التشغيل (Resting Voltage): {telemetry_summary.get('resting_voltage_v', 'N/A')} V
- أدنى جهد مسجل أثناء التدوير (Cranking Min Voltage): {telemetry_summary.get('min_cranking_voltage_v', 'N/A')} V
- مقدار هبوط الجهد عند التشغيل (Cranking Sag): {telemetry_summary.get('cranking_voltage_sag_v', 'N/A')} V
- متوسط شحن الدينامو أثناء عمل المحرك (Running Voltage Mean): {telemetry_summary.get('running_voltage_mean_v', 'N/A')} V
- هبوط الجهد عند تشغيل المكيف والأحمال (Load Voltage Drop): {telemetry_summary.get('load_voltage_drop_v', 'N/A')} V
- تموج وتذبذب الجهد (AC Ripple Noise): {telemetry_summary.get('voltage_ripple_mean_v', 'N/A')} V
- سرعة دوران المحرك (Engine RPM): {telemetry_summary.get('avg_engine_rpm', 'N/A')} RPM
- تصنيف نموذج الذكاء الاصطناعي (ML Prediction): {ml_prediction.get('predicted_class_ar', 'غير محدد')} (نسبة الثقة: {ml_prediction.get('confidence_pct', 95)}%)
- أكواد الأعطال (DTC): {dtc_text}

المطلوب صياغة تقرير تشخيصي هندسي متكامل ومباشر باللغة العربية يوضح:
1. التشخيص الدقيق للحالة والسبب الفيزيائي للخلل.
2. تقييم مستوى الخطورة (منخفض / متوسط / حرج).
3. هل يمكن متابعة القيادة حالياً؟
4. خطوات الفحص والصيانة الدقيقة الموصى بها.
"""
        # Execute live LLM API call
        success, response_or_err, latency = self._execute_chat([
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])

        if success:
            return {
                "source": "LIVE_LLM_API",
                "model_used": self.model_name,
                "latency_ms": latency,
                "advisory_text_ar": response_or_err,
                "status": "success",
            }
        else:
            # Inform user of exact API issue
            return {
                "source": "API_ERROR_NOTICE",
                "model_used": self.model_name,
                "latency_ms": latency,
                "advisory_text_ar": f"⚠️ **تنبيه مزود الذكاء الاصطناعي ({self.model_name}):**\n{response_or_err}\n\n*يمكنك ضبط مفتاح الـ API أو تغييره من زر الإعدادات ⚙️ بأعلى الصفحة.*",
                "status": "api_error",
            }

    def chat_with_mechanic(
        self,
        chat_history: List[Dict[str, str]],
        user_message: str,
        vehicle_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Interactive conversation with AI Mechanic via Live LLM API with live feedback.
        """
        context_str = ""
        if vehicle_context:
            context_str = f"\n[سياق القراءات اللحظية لسيارة المستخدم حالياً: {json.dumps(vehicle_context, ensure_ascii=False)}]\n"

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT + context_str}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})

        success, response_or_err, latency = self._execute_chat(messages)

        if success:
            return {
                "success": True,
                "source": "LIVE_LLM_API",
                "model": self.model_name,
                "latency_ms": latency,
                "response_ar": response_or_err,
            }
        else:
            # Return clear error notification
            error_msg = (
                f"⚠️ **تعذر استلام الرد المباشر من مزود الذكاء الاصطناعي ({self.model_name}):**\n\n"
                f"• **السبب:** {response_or_err}\n"
                f"• **الحل:** اضغط على زر **الإعدادات ⚙️** في أعلى يمين الصفحة، وتحقق من مفتاح الـ API الخاص بك أو جرب مزوداً آخر (مثل OpenAI أو Groq أو DeepSeek) مع اختبار الاتصال الفوري."
            )
            return {
                "success": False,
                "source": "API_ERROR",
                "model": self.model_name,
                "latency_ms": latency,
                "response_ar": error_msg,
            }

    def _execute_chat(self, messages: List[Dict[str, str]]) -> Tuple[bool, str, int]:
        """Performs HTTP POST to the LLM API and returns (success, text_or_error, latency_ms)."""
        if not self.api_key or len(self.api_key) < 5:
            return False, "مفتاح الـ API غير معين. يرجى إدخال المفتاح في الإعدادات ⚙️.", 0

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 800,
        }

        start_time = time.time()
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout_sec)
            latency = int((time.time() - start_time) * 1000)

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                return True, content, latency
            else:
                err_data = {}
                try:
                    err_data = resp.json()
                except Exception:
                    pass
                err_msg = err_data.get("error", {}).get("message", resp.text[:200]) if isinstance(err_data.get("error"), dict) else resp.text[:200]
                return False, f"خطأ من المزود (كود {resp.status_code}): {err_msg}", latency

        except requests.exceptions.Timeout:
            return False, "انتهت مهلة انتظار خادم الذكاء الاصطناعي (Timeout).", int((time.time() - start_time) * 1000)
        except Exception as e:
            return False, f"فشل الاتصال بالإنترنت أو الخادم: {str(e)}", int((time.time() - start_time) * 1000)
