"""
Smart OBD-II Electrical Diagnostic System
Module: LLM Diagnostic & Advisory Engine (المستشار الذكي لتشخيص كهرباء السيارات)

Integrates OpenAI-compatible Large Language Models (DeepSeek, OpenAI, Groq, etc.)
to provide conversational, explainable Arabic diagnostic insights and actionable mechanic advice.
Includes automated fallback to rule-based Arabic diagnostics when offline or on API timeout.
"""

import os
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class AutomotiveLLMAdvisor:
    """
    Intelligent Arabic Automotive Electrical Diagnostic Consultant powered by LLM.
    """

    SYSTEM_PROMPT = """أنت "خبير تشخيص كهرباء السيارات الذكي" (Smart OBD-II AI Mechanic) في نظام متقدم لمراقبة السيارات.
مهمتك: تحليل البيانات الكهربائية اللحظية للسيارة (جهد البطارية، شحن الدينامو، هبوط الجهد عند التشغيل، تذبذب التيار، والأعطال DTC) ومخرجات نماذج تعلم الآلة، ثم تقديم تقرير تشخيصي ونصائح واضحة باللغة العربية البسيطة والدقيقة هندسياً.

أسلوب الإجابة المطلوب:
1. الشرح بلغة عربية سلسة ومباشرة يفهمها قائد المركبة بدون تعقيدات أكاديمية مفرطة، مع ذكر المصطلحات الهندسية الشائعة بين قوسين.
2. توضيح مستوى الخطورة بوضوح (منخفض / متوسط / حرج).
3. تحديد هل قيادة السيارة آمنة حالياً أم تستوجب التوقف الفوري لتجنب انطفاء المحرك في الطريق أو تلف الحواسيب الإلكترونية (ECUs).
4. إعطاء خطوات فحص عملية مرتبة (فحص أصابع البطارية، شد السير، فحص الفيوزات، إلخ).
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_sec: int = 8,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")).rstrip("/")
        self.model_name = model_name or os.getenv("LLM_MODEL", "deepseek-chat")
        self.timeout_sec = timeout_sec

    def generate_advisory_report(
        self,
        telemetry_summary: Dict[str, Any],
        ml_prediction: Dict[str, Any],
        dtc_codes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive diagnostic report using LLM with graceful offline fallback.
        """
        dtc_text = ", ".join(dtc_codes) if dtc_codes else "لا توجد أكواد أعطال مسجلة (No DTCs)"
        
        prompt = f"""
تحليل حالة السيارة الكهربائية:
- الجهد اللحظي الحالي: {telemetry_summary.get('current_voltage', 'N/A')}V
- أدنى جهد أثناء التدوير (Cranking Min): {telemetry_summary.get('min_cranking_voltage', 'N/A')}V
- متوسط جهد الشحن والدينامو: {telemetry_summary.get('avg_running_voltage', 'N/A')}V
- مستوى تذبذب الجهد (Ripple): {telemetry_summary.get('voltage_ripple', 'N/A')}V
- حالة المحرك: {telemetry_summary.get('engine_state_desc', 'يعمل')}
- تشخيص نموذج الذكاء الاصطناعي: {ml_prediction.get('predicted_class_ar', 'غير محدد')} (نسبة الثقة: {ml_prediction.get('confidence_pct', 95)}%)
- أكواد الأعطال (DTC): {dtc_text}

المطلوب:
1. تشخيص المشكلة بدقة.
2. تقييم الخطورة (منخفض / متوسط / حرج).
3. هل يمكن متابعة القيادة؟
4. خطوات الصيانة والإصلاح المقترحة لسائق السيارة والميكانيكي.
"""
        # Try LLM API first
        llm_response = self._call_llm_api(prompt)
        if llm_response:
            return {
                "source": "LLM_AI_ADVISOR",
                "model_used": self.model_name,
                "advisory_text_ar": llm_response,
                "status": "success",
            }
            
        # If API is unreachable or invalid key, return high quality fallback
        fallback_text = self._generate_rule_based_fallback(telemetry_summary, ml_prediction)
        return {
            "source": "RULE_BASED_FALLBACK_ENGINE",
            "model_used": "Automotive Expert Rules v1.0",
            "advisory_text_ar": fallback_text,
            "status": "fallback_applied",
        }

    def chat_with_mechanic(
        self,
        chat_history: List[Dict[str, str]],
        user_message: str,
        vehicle_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Interactive chat allowing driver/technician to ask questions about the vehicle's electrical state.
        """
        context_str = ""
        if vehicle_context:
            context_str = f"\nسياق القراءات الحالية للسيارة:\n{vehicle_context}\n"
            
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT + context_str}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": user_message})
        
        response = self._call_llm_api_messages(messages)
        if response:
            return response
            
        # Offline / Fallback Intelligent Mechanic Responses based on user query & vehicle telemetry
        msg_lower = user_message.lower()
        curr_v = vehicle_context.get("running_voltage_mean_v", 14.1) if vehicle_context else 14.1
        crank_v = vehicle_context.get("min_cranking_voltage_v", 10.5) if vehicle_context else 10.5
        
        if "سفر" in user_message or "أمشي" in user_message or "أقدر" in user_message or "امشي" in user_message:
            if curr_v < 13.2:
                return (
                    f"⛔ **لا يُنصح بالسفر بالسيارة نهائياً بالوضع الحالي!**\n\n"
                    f"• **السبب:** جهد الدينامو منخفض جداً ({curr_v:.1f}V) مما يعني أن السيارة تسحب كل طاقتها من البطارية.\n"
                    f"• **النتيجة المحتملة:** بعد قطع مسافة 15 إلى 45 دقيقة ستنفد البطارية تماماً وينطفئ المحرك وتتوقف طلمبة البنزين وحواسيب الأمان.\n"
                    f"• **الحل:** توجه لأقرب كهربائي لإصلاح الدينامو وشده قبل السفر."
                )
            elif crank_v < 9.6:
                return (
                    f"⚠️ **يمكنك السفر ولكن بحذر شديد مع تجنب إطفاء المحرك:**\n\n"
                    f"• الجهد أثناء التدوير ({crank_v:.1f}V) ضعيف جداً، مما يعني أن البطارية متدهورة.\n"
                    f"• طالما المحرك يعمل، الدينامو يغذيه ({curr_v:.1f}V)، لكن إذا أطفأت السيارة في استراحة أو محطة وقود قد لا تدور مجدداً."
                )
            else:
                return "✅ **نعم، يمكنك متابعة السفر بأمان تام.** جميع القراءات الكهربائية الحالية للدينامو والبطارية في النطاق السليم والمثالي."

        elif "بطارية" in user_message or "دينامو" in user_message:
            return (
                f"🔍 **طريقة التمييز بين عطل البطارية والدينامو:**\n\n"
                f"1. **إذا كانت المشكلة من البطارية:**\n"
                f"   - السيارة تعاني في الصباح عند بدء التشغيل (صوت التدوير ثقيل وبطيء).\n"
                f"   - بمجرد دوران المحرك، الجهد يرتفع فوراً إلى 13.8V - 14.4V (الدينامو سليم).\n\n"
                f"2. **إذا كانت المشكلة من الدينامو:**\n"
                f"   - إضاءة لمبة البطارية الحمراء في التابلوه أثناء المشي.\n"
                f"   - ضعف إضاءة الأنوار عند تشغيل المكيف وهبوط الجهد تحت 13.2V أثناء عمل المحرك."
            )

        elif "خطوات" in user_message or "أفحص" in user_message or "افحص" in user_message or "بنفسي" in user_message or "diy" in msg_lower:
            return (
                "🛠️ **خطوات الفحص السريع بنفسك (DIY Checklist):**\n\n"
                "1. **فحص أصابع البطارية (Terminals):** تأكد من عدم وجود أملاح بيضاء أو زرقة، ونظفها بماء دافئ وبيكربونات صوديوم وأحكم ربطها.\n"
                "2. **فحص سير الدينامو (Alternator Belt):** تأكد من عدم وجود تشققات بالسير، ومن قوة شده (لا يرتخي أكثر من 1 سم عند الضغط عليه باصبعك).\n"
                "3. **فحص كابل التأريض (Chassis Ground):** تأكد أن السلك الأسود الواصل من البطارية لجسم السيارة مربوط بإحكام وبدون صدأ."
            )

        return (
            "👨‍🔧 **نصيحة الميكانيكي الذكي:**\n\n"
            f"بناءً على القراءات المسجلة حالياً (جهد التشغيل: {curr_v:.1f}V، جهد التدوير: {crank_v:.1f}V):\n"
            "احرص على مراقبة لمبة شحن البطارية بالتابلوه، وتأكد من ثبات الجهد عند تشغيل المكيف والأحمال العالية. يمكنك سؤالي عن أي استفسار آخر!"
        )


    def _call_llm_api(self, user_content: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self._call_llm_api_messages(messages)

    def _call_llm_api_messages(self, messages: List[Dict[str, str]]) -> Optional[str]:
        if not self.api_key or "your_api_key" in self.api_key:
            return None
            
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 600,
        }
        
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout_sec)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                return None
        except Exception:
            return None

    def _generate_rule_based_fallback(
        self,
        telemetry: Dict[str, Any],
        ml_prediction: Dict[str, Any],
    ) -> str:
        """High precision deterministic automotive electrical fallback generator."""
        label = ml_prediction.get("label", 0)
        curr_v = telemetry.get("current_voltage", 12.6)
        crank_v = telemetry.get("min_cranking_voltage", 10.5)
        
        if label == 0:
            return (
                "✅ **حالة النظام الكهربائي: سليمة ومثالية**\n\n"
                "• **التشخيص:** جهد البطارية وشحن الدينامو ضمن النطاق الهندسي القياسي.\n"
                "• **مستوى الخطورة:** منخفض (آمن تماماً).\n"
                "• **إمكانية القيادة:** يمكنك القيادة بأمان تام دون أي قلق.\n"
                "• **نصيحة وقائية:** حافظ على نظافة أقطاب البطارية وتفقد شد سير الدينامو في الصيانة الدورية."
            )
            
        elif label == 1:
            return (
                f"⚠️ **تحذير: ضعف وتدهور في كفاءة البطارية (هبوط الجهد أثناء التشغيل: {crank_v:.1f}V)**\n\n"
                "• **التشخيص:** هبوط الجهد أثناء بدء التشغيل عن الحد الأدنى الآمن (9.6V) يشير لارتفاع المقاومة الداخلية للبطارية أو قرب انتهاء عمرها الافتراضي.\n"
                "• **مستوى الخطورة:** متوسط إلى حرج في الأجواء الباردة.\n"
                "• **إمكانية القيادة:** السيارة تعمل حالياً عبر الدينامو، لكن قد تفشل في إعادة التشغيل إذا أطفأت المحرك.\n"
                "• **الإجراء الموصى به:**\n"
                "  1. تجنب إطفاء المحرك في الأماكن النائية.\n"
                "  2. افحص كابلات وأصابع البطارية من وجود كبرتة أو تآكل.\n"
                "  3. توجه لمركز فحص بطاريات لاختبار كفاءة الخلايا (CCA Test)."
            )
            
        elif label == 2:
            sub_desc = "شحن زائد خطير" if curr_v > 14.8 else "ضعف أو انعدام شحن الدينامو"
            return (
                f"🚨 **خطر: خلل في منظومة شحن الدينامو - {sub_desc} (الجهد الحالي: {curr_v:.1f}V)**\n\n"
                "• **التشخيص:** الدينامو لا يزود الشبكة الكهربائية بالجهد الصحيح، مما يجعل السيارة تستهلك طاقة البطارية مباشرة أو يعرض الحواسيب الإلكترونية لجهد مرتفع.\n"
                "• **مستوى الخطورة:** حرج جداً (Critical).\n"
                "• **إمكانية القيادة:** غير آمن! قد ينطفئ المحرك فجأة أثناء القيادة عند نفاد شحن البطارية.\n"
                "• **الإجراء الفوري:**\n"
                "  1. أطفئ فوراً جميع الأجهزة غير الضرورية (المكيف، الراديو، أنوار الضباب).\n"
                "  2. توجه مباشرة لأقرب فني كهرباء سيارات لفحص سير الدينامو ومنظم الجهد (Voltage Regulator)."
            )
            
        else:
            return (
                "⚡ **تنبيه: رصد تسريب كهربائي أو مقاومة تلامس وتأريض مرتفعة**\n\n"
                "• **التشخيص:** وجود هبوط ملحوظ في الجهد عند تشغيل الأجهزة أو استنزاف تيار غير طبيعي أثناء إيقاف تشغيل السيارة.\n"
                "• **مستوى الخطورة:** متوسط.\n"
                "• **إمكانية القيادة:** ممكنة ولكن مع الحذر من تفريغ البطارية عند ركن السيارة لفترات طويلة.\n"
                "• **الإجراء الموصى به:** فحص كابل التأريض الرئيسي (Chassis Ground) والتأكد من عدم وجود جهاز يستهلك كهرباء بعد إطفاء السيارة (مثل شاحن أو جهاز إنذار تالف)."
            )
