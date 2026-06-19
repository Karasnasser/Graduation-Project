"""
Smart Diet Planner - AI-Powered Personalized Nutritional Planning
================================================================

This module uses Google Gemini AI to generate medically accurate,
personalized 7-day meal plans for Kidney Disease patients based on
their KDIGO stage and lab results (eGFR, Potassium, Sodium, etc.).
"""

import os
import warnings
import google.generativeai as genai
from typing import Dict, Any, Optional
from config import GEMINI_API_KEY

# Suppress the FutureWarning from the deprecated google-generativeai package.
# The existing RAG module still uses it; we silence it here to keep output clean.
warnings.filterwarnings("ignore", category=FutureWarning, module="google")

class SmartDietPlanner:
    def __init__(self):
        """Initialize the Gemini model for nutritional planning."""
        self.api_key = GEMINI_API_KEY
        if self.api_key:
            genai.configure(api_key=self.api_key)
            # Use gemini-2.5-flash: best available model for medical text generation
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.is_active = True
        else:
            self.is_active = False
            print("[WARN] GEMINI_API_KEY is missing. Smart Diet Planner is disabled.")

    def _build_medical_prompt(self, patient_data: Dict[str, Any]) -> str:
        """
        Constructs a highly constrained medical prompt for Gemini based on
        KDIGO guidelines and the patient's specific lab anomalies.
        """
        # Extract patient data with defaults
        age = patient_data.get('age', 'غير محدد')
        weight = patient_data.get('weight', 70)
        egfr = patient_data.get('egfr', None)
        stage = patient_data.get('stage', 'Unknown')
        potassium = patient_data.get('potassium', 4.5)  # Normal 3.5-5.0
        sodium = patient_data.get('sodium', 140)        # Normal 135-145
        diabetes = patient_data.get('diabetes', 'no')
        hypertension = patient_data.get('hypertension', 'no')

        # Determine clinical rules based on KDIGO & Labs
        rules = []
        
        # 1. Protein Rule
        if egfr is not None:
            if egfr < 60: # Stage 3-5
                rules.append("1. **البروتين**: تقييد كمية البروتين إلى (0.6 - 0.8 جرام لكل كيلوجرام من وزن الجسم يومياً) لحماية الكلى المتبقية.")
            else:
                rules.append("1. **البروتين**: يمكن تناول كمية بروتين طبيعية (حوالي 1 جرام لكل كيلوجرام)، ويفضل البروتين النباتي.")
        
        # 2. Potassium Rule
        if potassium > 5.0:
            rules.append(f"2. **البوتاسيوم (تحذير عالي - {potassium} mEq/L)**: يمنع تماماً الأطعمة الغنية بالبوتاسيوم (الموز، الطماطم، البطاطس، البرتقال، التمر). اختر الفواكه قليلة البوتاسيوم مثل التفاح والتوت.")
        else:
            rules.append("2. **البوتاسيوم**: النسبة طبيعية، يمكن تناول الفواكه والخضروات باعتدال.")
            
        # 3. Sodium / BP Rule
        if sodium > 145 or str(hypertension).lower() in ['yes', '1', 'true']:
            rules.append("3. **الصوديوم**: تقييد صارم للملح (أقل من 2 جرام يومياً). يمنع تماماً المخللات، المعلبات، واللحوم المصنعة.")
        else:
            rules.append("3. **الصوديوم**: يفضل تقليل الملح في الطعام بشكل عام للحفاظ على ضغط الدم.")
            
        # 4. Diabetes Rule
        if str(diabetes).lower() in ['yes', '1', 'true']:
            rules.append("4. **السكري**: تجنب السكريات البسيطة والكربوهيدرات المكررة. التركيز على الحبوب الكاملة لضبط نسبة السكر في الدم.")

        rules_text = "\n".join(rules)

        prompt = f"""
أنت طبيب تغذية علاجية متخصص في أمراض الكلى (Renal Dietitian).
مهمتك هي إنشاء "نظام غذائي ذكي لمدة 7 أيام" لمريض يعاني من مرض الكلى المزمن بناءً على تحاليله الحالية.

[بيانات المريض]
- العمر: {age} سنة
- الوزن: {weight} كجم
- معدل الترشيح الكلوي (eGFR): {egfr if egfr else 'غير متوفر'}
- مرحلة الكلى (Stage): {stage}
- البوتاسيوم: {potassium} mEq/L
- الصوديوم: {sodium} mEq/L
- السكري: {'نعم' if str(diabetes).lower() in ['yes', '1', 'true'] else 'لا'}
- ضغط الدم: {'نعم' if str(hypertension).lower() in ['yes', '1', 'true'] else 'لا'}

[القواعد الطبية الصارمة التي يجب اتباعها في تصميم الوجبات]
{rules_text}

[المطلوب]
بناءً على القواعد السابقة حصراً، قم بتوليد تقرير باللغة العربية بتنسيق Markdown يحتوي على:
1. **ملخص الأهداف الغذائية اليومية:** (كمية البروتين، الصوديوم، السعرات المسموحة بناءً على وزنه).
2. **قائمة الأطعمة الممنوعة تماماً:** (بناءً على تحاليله المذكورة أعلاه).
3. **نظام غذائي مفصل لمدة 7 أيام:** يتضمن (الإفطار، الغداء، العشاء، وجبة خفيفة) لكل يوم. الوجبات يجب أن تكون من ثقافة الطعام العربية/المصرية (مثال: جبن قريش بدون ملح، خبز أسمر، دجاج مشوي، سلطة بدون طماطم إذا كان البوتاسيوم مرتفع).
4. **نصائح عامة للترطيب وشرب الماء.**

تأكد أن الوجبات لذيذة وواقعية، ولكنها تلتزم 100% بالقيود الطبية الصارمة المذكورة أعلاه، خاصة البوتاسيوم والبروتين!
لا تضع مقدمات طويلة، ابدأ بالتقرير فوراً باستخدام العناوين (#).
"""
        return prompt

    def generate_diet_plan(self, patient_data: Dict[str, Any]) -> str:
        """
        Calls Gemini to generate the personalized diet plan.
        """
        if not self.is_active:
            return "عذراً، خدمة النظام الغذائي الذكي غير مفعلة لعدم وجود مفتاح GEMINI_API_KEY."
            
        try:
            prompt = self._build_medical_prompt(patient_data)
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,        # Low temperature = medically consistent
                    max_output_tokens=8192, # Full 7-day plan needs room
                ),
                # Disable internal thinking so all tokens go to the answer
                request_options=genai.types.RequestOptions(timeout=120)
            )
            return response.text
        except Exception as e:
            return f"حدث خطأ أثناء توليد النظام الغذائي: {str(e)}"
