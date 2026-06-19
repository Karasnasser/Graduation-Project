"""
RAG (Retrieval-Augmented Generation) Module
Medical knowledge base with Gemini LLM for kidney disease Q&A.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Load .env file so GEMINI_API_KEY is available
from dotenv import load_dotenv
load_dotenv()


# Vector database
import chromadb
from chromadb.config import Settings

# Google Gemini
import google.generativeai as genai


@dataclass
class RetrievedDocument:
    """Retrieved document from knowledge base."""
    content: str
    source: str
    relevance_score: float


class KidneyKnowledgeBase:
    """
    Knowledge base for kidney disease information.
    Uses ChromaDB for vector storage and retrieval.
    """
    
    # Medical knowledge about kidney disease
    KIDNEY_KNOWLEDGE = [
        {
            "id": "ckd_overview",
            "content": """
            Chronic Kidney Disease (CKD) is a condition where the kidneys are damaged and cannot filter blood properly.
            مرض الكلى المزمن هو حالة تتضرر فيها الكلى ولا تستطيع تصفية الدم بشكل صحيح.
            
            Key facts:
            - CKD affects about 10% of the world's population
            - Early detection can slow progression
            - Main causes: diabetes (40%) and hypertension (25%)
            - Often has no symptoms in early stages
            """,
            "source": "CKD Overview - KDIGO Guidelines"
        },
        {
            "id": "gfr_stages",
            "content": """
            GFR Stages of Chronic Kidney Disease:
            مراحل مرض الكلى المزمن حسب معدل الترشيح الكبيبي:
            
            - G1: GFR ≥ 90 - Normal or high kidney function (وظائف الكلى طبيعية)
            - G2: GFR 60-89 - Mildly decreased function (انخفاض طفيف)
            - G3a: GFR 45-59 - Mild to moderate decrease (انخفاض طفيف إلى متوسط)
            - G3b: GFR 30-44 - Moderate to severe decrease (انخفاض متوسط إلى شديد)
            - G4: GFR 15-29 - Severe decrease (انخفاض شديد)
            - G5: GFR < 15 - Kidney failure requiring dialysis (فشل كلوي)
            """,
            "source": "KDIGO CKD Staging Guidelines 2012"
        },
        {
            "id": "albuminuria",
            "content": """
            Albuminuria Categories (ACR - Albumin to Creatinine Ratio):
            فئات البروتين في البول:
            
            - A1: ACR < 30 mg/g - Normal to mildly increased (طبيعي)
            - A2: ACR 30-300 mg/g - Moderately increased / Microalbuminuria (مرتفع بشكل متوسط)
            - A3: ACR > 300 mg/g - Severely increased / Macroalbuminuria (مرتفع بشكل شديد)
            
            High albumin in urine indicates kidney damage and increases cardiovascular risk.
            ارتفاع البروتين في البول يشير إلى تلف الكلى ويزيد من خطر أمراض القلب.
            """,
            "source": "KDIGO Albuminuria Classification"
        },
        {
            "id": "egfr_calculation",
            "content": """
            eGFR Calculation (CKD-EPI 2021 Equation):
            حساب معدل الترشيح الكبيبي المقدر:
            
            The CKD-EPI equation uses serum creatinine, age, and sex to estimate GFR.
            معادلة CKD-EPI تستخدم الكرياتينين في الدم والعمر والجنس لتقدير GFR.
            
            Normal eGFR is above 90 mL/min/1.73m².
            المعدل الطبيعي أعلى من 90 مل/دقيقة/1.73م².
            
            Factors affecting eGFR:
            - Age: eGFR naturally decreases with age
            - Muscle mass: affects creatinine levels
            - Diet: high protein intake increases creatinine
            - Medications: some drugs affect kidney function
            """,
            "source": "CKD-EPI Creatinine Equation 2021"
        },
        {
            "id": "creatinine_info",
            "content": """
            Serum Creatinine (الكرياتينين في الدم):
            
            - Normal range: 0.5-1.5 mg/dL (varies by age, sex, muscle mass)
            - المعدل الطبيعي: 0.5-1.5 ملجم/ديسيلتر
            
            High creatinine indicates:
            - Reduced kidney function
            - Dehydration
            - High protein diet
            - Certain medications
            
            ارتفاع الكرياتينين يعني:
            - انخفاض وظائف الكلى
            - الجفاف
            - نظام غذائي عالي البروتين
            - بعض الأدوية
            """,
            "source": "Lab Values Reference"
        },
        {
            "id": "diabetic_nephropathy",
            "content": """
            Diabetic Nephropathy (اعتلال الكلى السكري):
            
            Diabetic nephropathy is kidney disease caused by diabetes.
            اعتلال الكلى السكري هو مرض كلوي يسببه مرض السكري.
            
            Stages:
            1. Hyperfiltration (early stage)
            2. Microalbuminuria (earliest clinical sign)
            3. Macroalbuminuria
            4. Progressive decline in GFR
            5. End-stage renal disease (ESRD)
            
            Prevention:
            - Good blood sugar control (HbA1c < 7%)
            - Blood pressure control (<130/80 mmHg)
            - ACE inhibitors or ARBs
            - Regular kidney function monitoring
            """,
            "source": "ADA Diabetic Kidney Disease Guidelines"
        },
        {
            "id": "risk_factors",
            "content": """
            Risk Factors for CKD (عوامل خطر مرض الكلى المزمن):
            
            Major risk factors:
            - Diabetes (السكري) - #1 cause
            - Hypertension (ارتفاع ضغط الدم) - #2 cause
            - Family history of kidney disease
            - Age > 60 years
            - Obesity
            - Smoking
            - Cardiovascular disease
            
            Controllable factors:
            - Blood pressure control
            - Blood sugar control
            - Healthy diet (low salt, adequate protein)
            - Regular exercise
            - Avoid NSAIDs
            - Stay hydrated
            """,
            "source": "CKD Risk Factor Guidelines"
        },
        {
            "id": "symptoms",
            "content": """
            CKD Symptoms (أعراض مرض الكلى المزمن):
            
            Early stages (G1-G3a): Usually NO symptoms
            المراحل المبكرة: عادة لا توجد أعراض
            
            Later stages symptoms:
            - Fatigue and weakness (تعب وضعف)
            - Swelling in legs, ankles, feet (تورم)
            - Shortness of breath (ضيق التنفس)
            - Nausea and vomiting (غثيان وقيء)
            - Loss of appetite (فقدان الشهية)
            - Itching (حكة)
            - Changes in urination (تغيرات في التبول)
            - Muscle cramps (تشنجات عضلية)
            
            This is why regular screening is important!
            """,
            "source": "CKD Clinical Manifestations"
        },
        {
            "id": "treatment_options",
            "content": """
            CKD Treatment Options (خيارات علاج مرض الكلى):
            
            Conservative Management (for stages G1-G4):
            - Blood pressure control (target <130/80)
            - Blood sugar control (HbA1c < 7%)
            - SGLT2 inhibitors (new medications that protect kidneys)
            - ACE inhibitors/ARBs
            - Dietary modifications (low salt, controlled protein)
            - Avoid nephrotoxic drugs
            
            Renal Replacement Therapy (for stage G5):
            - Hemodialysis (غسيل الكلى)
            - Peritoneal dialysis (غسيل بريتوني)
            - Kidney transplant (زراعة الكلى)
            
            التشخيص المبكر يمكن أن يؤخر أو يمنع الحاجة للغسيل الكلوي.
            """,
            "source": "KDIGO Treatment Guidelines"
        },
        {
            "id": "diet_recommendations",
            "content": """
            Diet for Kidney Disease (نظام غذائي لمرض الكلى):
            
            General recommendations:
            - Reduce sodium/salt (< 2g/day) - تقليل الملح
            - Control protein intake (0.6-0.8 g/kg body weight)
            - Limit potassium (if levels are high) - تقليل البوتاسيوم
            - Limit phosphorus - تقليل الفوسفور
            - Adequate hydration (unless fluid restricted)
            
            Foods to avoid:
            - Processed foods (high sodium)
            - Bananas, oranges (high potassium)
            - Dairy products (high phosphorus)
            - Red meat (high protein, phosphorus)
            
            Foods recommended:
            - Fresh fruits (apples, berries)
            - Fresh vegetables (cabbage, peppers)
            - Fish (omega-3)
            - Olive oil
            """,
            "source": "Renal Diet Guidelines"
        }
    ]
    
    def __init__(self, persist_dir: str = "knowledge_base"):
        """Initialize knowledge base with ChromaDB."""
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="kidney_knowledge",
            metadata={"description": "Medical knowledge about kidney disease"}
        )
        
        # Load knowledge if empty
        if self.collection.count() == 0:
            self._load_knowledge()
    
    def _load_knowledge(self):
        """Load medical knowledge into ChromaDB."""
        print(" Loading medical knowledge base...")
        
        documents = []
        metadatas = []
        ids = []
        
        for item in self.KIDNEY_KNOWLEDGE:
            documents.append(item["content"])
            metadatas.append({"source": item["source"]})
            ids.append(item["id"])
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"[OK] Loaded {len(documents)} knowledge documents")
    
    def search(
        self,
        query: str,
        n_results: int = 3
    ) -> List[RetrievedDocument]:
        """Search knowledge base for relevant documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        documents = []
        for i, doc in enumerate(results['documents'][0]):
            documents.append(RetrievedDocument(
                content=doc,
                source=results['metadatas'][0][i].get('source', 'Unknown'),
                relevance_score=1 - results['distances'][0][i] if results['distances'] else 0.5
            ))
        
        return documents
    
    def add_document(self, doc_id: str, content: str, source: str):
        """Add new document to knowledge base."""
        self.collection.add(
            documents=[content],
            metadatas=[{"source": source}],
            ids=[doc_id]
        )


class GeminiRAG:
    """
    RAG system using Google Gemini for kidney disease Q&A.
    """
    
    SYSTEM_PROMPT = """You are a medical AI assistant specialized in kidney disease.
    
    Your role:
    - Answer questions about kidney disease, CKD stages, and lab values
    - Explain medical terms in simple language
    - Provide information from the retrieved medical documents
    - Always remind users to consult their doctor for medical decisions
    
    Guidelines:
    - Be accurate and cite the sources when available
    - Be compassionate and supportive
    - If you don't know something, say so
    - Never prescribe medications or give specific medical advice
    
    IMPORTANT: Always include a disclaimer that this is for informational purposes only.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini RAG.
        
        Args:
            api_key: Google Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
        """
        # Get API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            print("[WARN] Warning: No Gemini API key provided. Set GEMINI_API_KEY environment variable.")
            self.model = None
        else:
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize knowledge base
        self.knowledge_base = KidneyKnowledgeBase()
        
        # Conversation history
        self.history = []
    
    def ask(
        self,
        question: str,
        patient_context: Optional[Dict[str, Any]] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question about kidney disease.
        
        Args:
            question: User's question
            patient_context: Optional patient info (egfr, stage, etc.)
            include_sources: Whether to include source citations
            
        Returns:
            Dictionary with answer and sources
        """
        if self.model is None:
            return {
                "answer": "❌ Gemini API key not configured. Please set GEMINI_API_KEY environment variable.",
                "sources": [],
                "error": True
            }
        
        # Retrieve relevant documents
        retrieved_docs = self.knowledge_base.search(question, n_results=3)
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for doc in retrieved_docs:
            context_parts.append(f"[Source: {doc.source}]\n{doc.content}")
            sources.append({
                "source": doc.source,
                "relevance": round(doc.relevance_score, 2)
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build patient context if provided
        patient_info = ""
        if patient_context:
            patient_info = f"""
            Patient Information:
            - eGFR: {patient_context.get('egfr', 'N/A')} mL/min/1.73m²
            - GFR Stage: {patient_context.get('gfr_stage', 'N/A')}
            - ACR: {patient_context.get('acr', 'N/A')} mg/g
            - Risk Level: {patient_context.get('risk_level', 'N/A')}
            """
        
        # Build prompt
        prompt = f"""{self.SYSTEM_PROMPT}

{patient_info}

Retrieved Medical Information:
{context}

---

User Question: {question}

Please provide a helpful, accurate answer based on the retrieved information.
End with a reminder to consult a doctor.
"""
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            answer = response.text
            
            # Add to history
            self.history.append({
                "question": question,
                "answer": answer
            })
            
            result = {
                "answer": answer,
                "sources": sources if include_sources else [],
                "error": False
            }
            
            return result
            
        except Exception as e:
            return {
                "answer": f"❌ Error generating response: {str(e)}",
                "sources": sources,
                "error": True
            }
    
    def explain_result(
        self,
        egfr: float,
        gfr_stage: str,
        acr: Optional[float] = None,
        risk_level: str = None
    ) -> str:
        """
        Generate explanation for lab results.
        
        Args:
            egfr: eGFR value
            gfr_stage: GFR stage (G1-G5)
            acr: ACR value if available
            risk_level: Risk level
            
        Returns:
            Explanation text
        """
        patient_context = {
            "egfr": egfr,
            "gfr_stage": gfr_stage,
            "acr": acr,
            "risk_level": risk_level
        }
        
        question = f"""
        Please explain what my kidney test results mean:
        - eGFR: {egfr} mL/min/1.73m²
        - Stage: {gfr_stage}
        {"- ACR: " + str(acr) + " mg/g" if acr else ""}
        {"- Risk Level: " + risk_level if risk_level else ""}
        
        What does this mean for my health? What should I do next?
        """
        
        result = self.ask(question, patient_context)
        return result["answer"] if not result["error"] else "Could not generate explanation."
    
    def get_recommendations(
        self,
        gfr_stage: str,
        has_diabetes: bool = False,
        has_hypertension: bool = False
    ) -> str:
        """Get personalized recommendations based on patient condition."""
        conditions = []
        if has_diabetes:
            conditions.append("diabetes")
        if has_hypertension:
            conditions.append("hypertension")
        
        question = f"""
        I have kidney disease stage {gfr_stage}.
        {"I also have: " + ", ".join(conditions) if conditions else ""}
        
        What lifestyle changes and treatments are recommended for my condition?
        Please provide practical advice I can follow.
        """
        
        result = self.ask(question)
        return result["answer"] if not result["error"] else "Could not generate recommendations."
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []


if __name__ == "__main__":
    # Test the RAG system
    print("Testing Kidney Disease RAG System")
    print("=" * 50)
    
    # Initialize (will use env variable for API key)
    rag = GeminiRAG()
    
    # Test knowledge base search
    print("\n Testing Knowledge Base Search:")
    docs = rag.knowledge_base.search("What is creatinine?")
    for doc in docs:
        print(f"  - {doc.source} (relevance: {doc.relevance_score:.2f})")
    
    # Test question (only if API key is set)
    if rag.model:
        print("\n Testing Question Answering:")
        result = rag.ask("What does eGFR 28 mean?")
        print(result["answer"][:500] + "..." if len(result["answer"]) > 500 else result["answer"])
