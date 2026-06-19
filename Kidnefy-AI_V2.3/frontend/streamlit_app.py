import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Kidnefy AI — Kidney Disease Platform",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Cairo', 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1117 0%, #1a1f2e 100%); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #00d4aa);
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    padding: 12px 24px !important; transition: all 0.25s !important;
    font-family: 'Cairo', sans-serif !important;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(79,142,247,0.3) !important; }
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(79,142,247,0.08), rgba(0,212,170,0.05));
    border: 1px solid rgba(79,142,247,0.2); border-radius: 14px; padding: 16px;
}
.stAlert { border-radius: 10px !important; }
.stProgress > div > div { background: linear-gradient(90deg, #4f8ef7, #00d4aa) !important; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Kidnefy AI")
    st.markdown("---")
    page = st.selectbox("Navigate to:", [
        "🔬 AI Analysis",
        "🫁 CT Scan Classifier",
        "📊 KDIGO Staging",
        "🥗 Diet Planner",
        "🔔 Smart Alerts",
        "🔄 What-If Simulator",
        "💬 AI Assistant",
        "📋 PDF Report",
    ])
    st.markdown("---")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        if r.ok:
            st.success("✅ API Online")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
    st.caption(f"`{API_BASE}`")

# ── HELPERS ──────────────────────────────────────────────
def api_post(endpoint, json_data=None, files=None):
    try:
        if files:
            r = requests.post(f"{API_BASE}{endpoint}", files=files, timeout=30)
        else:
            r = requests.post(f"{API_BASE}{endpoint}", json=json_data, timeout=30)
        if r.ok:
            return r.json(), None
        return None, r.json().get("detail", f"Error {r.status_code}")
    except Exception as e:
        return None, str(e)

def risk_color(risk: str):
    if "Low" in risk: return "🟢"
    if "Moderate" in risk: return "🟡"
    if "High" in risk: return "🔴"
    return "⚫"

# ══════════════════════════════════════════════════════════
# PAGE 1 — AI Analysis
# ══════════════════════════════════════════════════════════
if page == "🔬 AI Analysis":
    st.title("🔬 CKD Prediction & Risk Analysis")
    st.caption("Predict Chronic Kidney Disease from lab values using an Ensemble AI model (XGBoost + Neural Network).")

    with st.expander("📋 Load Sample Patient", expanded=False):
        preset = st.selectbox("Sample", ["Advanced CKD (G4)", "Healthy", "Mild CKD (G2/G3)"])
        presets = {
            "Advanced CKD (G4)": dict(name="Ahmed Ali", age=70, sex="male", cr=2.3, uacr=44.0, urea=61, hemo=11.0),
            "Healthy":           dict(name="Sara Mahmoud", age=35, sex="female", cr=0.8, uacr=5.0, urea=15, hemo=13.5),
            "Mild CKD (G2/G3)":  dict(name="Omar Hassan", age=55, sex="male", cr=1.5, uacr=80.0, urea=30, hemo=12.0),
        }
        if st.button("Load"):
            for k, v in presets[preset].items():
                st.session_state[f"a_{k}"] = v

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Patient Name", st.session_state.get("a_name", ""))
        age  = st.number_input("Age", 1, 120, int(st.session_state.get("a_age", 55)))
        sex  = st.selectbox("Sex", ["male", "female"], index=0 if st.session_state.get("a_sex","male")=="male" else 1)
        cr   = st.number_input("Creatinine (mg/dL)", 0.1, 20.0, float(st.session_state.get("a_cr", 1.5)), 0.1)
    with col2:
        uacr = st.number_input("UACR (mg/g)", 0.0, 5000.0, float(st.session_state.get("a_uacr", 30.0)), 5.0)
        urea = st.number_input("Blood Urea (mg/dL)", 1.0, 300.0, float(st.session_state.get("a_urea", 25.0)), 1.0)
        hemo = st.number_input("Hemoglobin (g/dL)", 1.0, 25.0, float(st.session_state.get("a_hemo", 13.0)), 0.5)
        bp   = st.number_input("Systolic BP (mmHg)", 50, 250, 120, 5)

    if st.button("🔬 Run AI Analysis", use_container_width=True, type="primary"):
        payload = {"patient": {"name": name, "age": age, "sex": sex},
                   "lab_values": {"creatinine": cr, "acr": uacr, "blood_urea": urea, "hemoglobin": hemo}}
        with st.spinner("Analyzing with Ensemble AI..."):
            data, err = api_post("/predict", payload)
        if err:
            st.error(f"Error: {err}")
        else:
            st.success("✅ Analysis Complete!")
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("CKD Prediction", "Positive ⚠️" if data.get("prediction") else "Negative ✅")
            m2.metric("Probability", f"{data.get('probability',0)*100:.1f}%")
            m3.metric("eGFR", f"{data.get('egfr',0):.1f}")
            m4.metric("Stage", data.get("gfr_stage","—"))
            risk = data.get("risk_level","—")
            st.info(f"{risk_color(risk)} **Risk Level:** {risk}  |  **Progression Risk:** {data.get('progression_risk_percent',0):.1f}%")
            if data.get("alerts"):
                for a in data["alerts"]: st.warning(a)
            if data.get("recommendations"):
                with st.expander("📋 Medical Recommendations"):
                    for r in data["recommendations"]: st.markdown(f"- {r}")

# ══════════════════════════════════════════════════════════
# PAGE 3 — CT Scan Classifier
# ══════════════════════════════════════════════════════════
elif page == "🫁 CT Scan Classifier":
    st.title("🫁 CT Scan Kidney Classifier")
    st.caption("Upload a kidney CT scan. MobileNetV2 will classify it as **Normal**, **Cyst**, **Stone**, or **Tumor**.")
    st.info("ℹ️ This is a clinical decision-support tool. Always confirm with a radiologist.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload Kidney CT Scan", type=["jpg","jpeg","png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded CT Scan", use_column_width=True)

    with col2:
        if uploaded:
            if st.button("🔍 Analyze CT Scan", use_container_width=True, type="primary"):
                with st.spinner("Analyzing with MobileNetV2 CNN..."):
                    data, err = api_post("/predict/ct", files={"file": ("ct.jpg", uploaded.getvalue(), "image/jpeg")})
                if err:
                    st.error(f"CT Analysis Error: {err}")
                else:
                    label = data.get("prediction","Unknown")
                    conf  = data.get("confidence_percentage", data.get("confidence",0)*100 if data.get("confidence") else 0)
                    icons = {"Normal":"✅","Cyst":"🫧","Stone":"🪨","Tumor":"⚠️"}
                    colors= {"Normal":"success","Cyst":"info","Stone":"warning","Tumor":"error"}
                    icon  = icons.get(label,"❓")
                    
                    st.markdown(f"### {icon} Classification: **{label}**")
                    st.metric("Confidence", f"{conf:.1f}%")
                    
                    probs = data.get("class_probabilities",{})
                    if probs:
                        st.subheader("Confidence Breakdown")
                        color_map = {"Normal":"#22c55e","Cyst":"#3b82f6","Stone":"#f59e0b","Tumor":"#ef4444"}
                        for cls, prob in probs.items():
                            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
                    
                    note = data.get("clinical_note","")
                    if note:
                        st.info(f"🩺 **Clinical Note:** {note}")
        else:
            st.markdown("""
            <div style='text-align:center; padding:80px 20px; opacity:0.5;'>
                <div style='font-size:72px;'>🫁</div>
                <h3>Upload a CT scan to begin</h3>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 — KDIGO Staging
# ══════════════════════════════════════════════════════════
elif page == "📊 KDIGO Staging":
    st.title("📊 KDIGO Staging & eGFR Calculator")
    st.caption("Calculate eGFR using CKD-EPI 2021 and determine disease stage per KDIGO guidelines.")
    col1, col2 = st.columns(2)
    with col1:
        cr  = st.number_input("Creatinine (mg/dL)", 0.1, 20.0, 1.5, 0.1)
        age = st.number_input("Age", 1, 120, 55)
        sex = st.selectbox("Sex", ["male","female"])
    with col2:
        acr = st.number_input("ACR (mg/g)", 0.0, 5000.0, 30.0, 5.0)
        
    if st.button("📊 Calculate Stage", use_container_width=True, type="primary"):
        with st.spinner("Calculating..."):
            data, err = api_post("/stage", {"creatinine": cr, "age": age, "acr": acr, "is_female": sex=="female"})
        if err:
            st.error(err)
        else:
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("eGFR", f"{data.get('egfr',0):.1f} mL/min/1.73m²")
            m2.metric("GFR Stage", data.get("gfr_stage","—"))
            m3.metric("Albuminuria", data.get("albuminuria_category","—"))
            m4.metric("Risk", data.get("risk_level","—"))
            if data.get("description"): st.info(data["description"])
            if data.get("recommendations"):
                with st.expander("📋 Recommendations"):
                    for r in data["recommendations"]: st.markdown(f"- {r}")

# ══════════════════════════════════════════════════════════
# PAGE 5 — Diet Planner
# ══════════════════════════════════════════════════════════
elif page == "🥗 Diet Planner":
    st.title("🥗 Smart Diet Planner")
    st.caption("Generate a personalized 7-day kidney-safe meal plan based on KDIGO nutritional guidelines.")

    col1, col2 = st.columns(2)
    with col1:
        stage    = st.selectbox("CKD Stage", ["G1","G2","G3a","G3b","G4","G5"], index=3)
        age      = st.number_input("Age", 18, 100, 60)
        weight   = st.number_input("Weight (kg)", 40.0, 200.0, 80.0, 0.5)
        diabetes = st.checkbox("Has Diabetes")
    with col2:
        potassium  = st.number_input("Potassium (mmol/L)", 2.0, 8.0, 5.2, 0.1)
        phosphorus = st.number_input("Phosphorus (mg/dL)", 1.0, 10.0, 4.5, 0.1)
        glucose    = st.number_input("Blood Glucose (mg/dL)", 50, 500, 110)
        sodium     = st.number_input("Sodium (mmol/L)", 100, 160, 138)

    if st.button("🥗 Generate 7-Day Diet Plan", use_container_width=True, type="primary"):
        payload = {
            "ckd_stage": stage, "age": age, "weight_kg": weight,
            "potassium_level": potassium, "phosphorus_level": phosphorus,
            "glucose_level": glucose, "sodium_level": sodium,
            "has_diabetes": diabetes
        }
        with st.spinner("Generating personalized plan with Gemini AI..."):
            data, err = api_post("/diet/plan", payload)
        if err:
            st.error(f"Diet Planner Error: {err}")
        else:
            restrictions = data.get("dietary_restrictions",[])
            if restrictions:
                st.warning("⚠️ **Key Restrictions:** " + " | ".join(restrictions))
            
            days = data.get("diet_plan",{}).get("days", data.get("days",[]))
            if days:
                day_names = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
                tabs = st.tabs([f"Day {d.get('day',i+1)} — {day_names[i%7]}" for i,d in enumerate(days)])
                for tab, day in zip(tabs, days):
                    with tab:
                        for meal in day.get("meals",[]):
                            mtype = meal.get("type", meal.get("meal_type",""))
                            food  = meal.get("food", meal.get("description",""))
                            notes = meal.get("reason", meal.get("notes",""))
                            with st.expander(f"🍽️ **{mtype}** — {food}"):
                                if notes: st.caption(notes)
            else:
                st.json(data)

# ══════════════════════════════════════════════════════════
# PAGE 6 — Smart Alerts
# ══════════════════════════════════════════════════════════
elif page == "🔔 Smart Alerts":
    st.title("🔔 Smart Alerts & Monitoring")
    st.caption("NLP symptom analysis and anomaly detection for patient monitoring.")

    tab1, tab2 = st.tabs(["🗣️ NLP Symptom Analysis", "📡 Anomaly Detection"])
    with tab1:
        symptoms = st.text_area("Patient Complaint (Arabic or English)", 
            "أشعر بتورم شديد في قدمي ولا أستطيع التبول منذ الصباح", height=120)
        patient_id = st.text_input("Patient ID", "demo_patient_001")
        if st.button("Analyze Symptoms", type="primary"):
            with st.spinner("Analyzing..."):
                data, err = api_post("/alerts/analyze", {"text": symptoms, "patient_id": patient_id})
            if err:
                st.error(err)
            else:
                urgency = data.get("urgency","—")
                icons = {"CRITICAL":"🚨","WARNING":"⚠️","INFO":"ℹ️"}
                st.markdown(f"### {icons.get(urgency,'❓')} Urgency: **{urgency}**")
                if data.get("possible_conditions"):
                    st.write("**Possible Conditions:**", ", ".join(data["possible_conditions"]))
                if data.get("recommendations"):
                    for r in data["recommendations"]: st.markdown(f"- {r}")
    with tab2:
        if st.button("📡 Scan for Anomalies", type="primary"):
            with st.spinner("Running Isolation Forest..."):
                data, err = api_post("/alerts/patient/demo_patient_001")
            if err:
                st.error(err)
            else:
                alerts = data.get("alerts",[]) if isinstance(data, dict) else data
                if alerts:
                    for alert in alerts:
                        sev = alert.get("severity","INFO")
                        if sev == "CRITICAL": st.error(f"🚨 **{alert.get('title','Alert')}** — {alert.get('message','')}")
                        elif sev == "WARNING": st.warning(f"⚠️ **{alert.get('title','Alert')}** — {alert.get('message','')}")
                        else: st.info(f"ℹ️ **{alert.get('title','Alert')}** — {alert.get('message','')}")
                else:
                    st.success("No anomalies detected.")

# ══════════════════════════════════════════════════════════
# PAGE 7 — What-If Simulator
# ══════════════════════════════════════════════════════════
elif page == "🔄 What-If Simulator":
    st.title("🔄 What-If Treatment Simulator")
    st.caption("Simulate how improving clinical parameters affects CKD progression risk.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Baseline (Current)")
        b_cr   = st.number_input("Creatinine — Baseline", 0.5, 8.0, 2.5, 0.1, key="b_cr")
        b_bp   = st.number_input("Systolic BP — Baseline", 90, 220, 160, key="b_bp")
        b_uacr = st.number_input("UACR — Baseline", 0, 1000, 200, key="b_uacr")
    with col2:
        st.subheader("Target (After Treatment)")
        t_cr   = st.number_input("Creatinine — Target", 0.5, 8.0, 1.8, 0.1, key="t_cr")
        t_bp   = st.number_input("Systolic BP — Target", 90, 220, 125, key="t_bp")
        t_uacr = st.number_input("UACR — Target", 0, 1000, 30, key="t_uacr")

    age = st.number_input("Age", 1, 100, 65)
    sex = st.selectbox("Sex", ["male","female"])

    if st.button("⚙️ Run Simulation", use_container_width=True, type="primary"):
        def uacr_to_al(u): return 0 if u < 30 else (2 if u < 300 else 4)
        payload = {
            "baseline": {"age": age, "sex": sex, "sc": b_cr, "bp": b_bp, "al": uacr_to_al(b_uacr), "dm": "no"},
            "modified": {"age": age, "sex": sex, "sc": t_cr, "bp": t_bp, "al": uacr_to_al(t_uacr), "dm": "no"}
        }
        with st.spinner("Running simulation..."):
            data, err = api_post("/predict/whatif", payload)
        if err:
            st.error(err)
        else:
            base = data.get("baseline",{})
            sim  = data.get("modified",{})
            deltas = data.get("deltas",{})
            
            st.subheader("Simulation Results")
            c1,c2,c3 = st.columns(3)
            c1.metric("Baseline Risk", f"{base.get('progression_risk_percent',0):.1f}%")
            c2.metric("Target Risk",   f"{sim.get('progression_risk_percent',0):.1f}%",
                      delta=f"{deltas.get('progression_risk',0):.1f}%")
            c3.metric("eGFR Change",   f"{sim.get('egfr',0):.1f}",
                      delta=f"{deltas.get('egfr',0):+.1f}")
            
            if deltas.get("risk_improved"):
                st.success(f"✅ Treatment reduces progression risk by {abs(deltas.get('probability',0)*100):.1f}%")
            else:
                st.warning("⚠️ This scenario does not improve risk. Adjust targets.")
            if data.get("impact_summary"):
                st.info(data["impact_summary"])

# ══════════════════════════════════════════════════════════
# PAGE 8 — AI Assistant
# ══════════════════════════════════════════════════════════
elif page == "💬 AI Assistant":
    st.title("💬 Medical AI Assistant (RAG)")
    st.caption("Ask questions about kidney disease — powered by Gemini + ChromaDB medical knowledge base.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "مرحباً! أنا المساعد الطبي الذكي لأمراض الكلى. كيف يمكنني مساعدتك اليوم؟"}
        ]

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🫀" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])

    if prompt := st.chat_input("اكتب سؤالك هنا..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🫀"):
            with st.spinner("Thinking..."):
                data, err = api_post("/chat", {"question": prompt})
            if err:
                answer = f"عذراً، حدث خطأ: {err}"
            else:
                answer = data.get("answer","No response.")
            st.markdown(answer)
            st.session_state.chat_history.append({"role":"assistant","content": answer})
            if data and data.get("sources"):
                with st.expander("📚 Sources"):
                    for s in data["sources"]: st.caption(f"- {s.get('source','')}")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("ما معنى eGFR 28؟"): st.session_state.setdefault("_q","ما معنى eGFR 28؟")
    with col_b:
        if st.button("ما هي إرشادات KDIGO؟"): st.session_state.setdefault("_q","ما هي إرشادات KDIGO؟")

# ══════════════════════════════════════════════════════════
# PAGE 9 — PDF Report
# ══════════════════════════════════════════════════════════
elif page == "📋 PDF Report":
    st.title("📋 Medical PDF Report Generator")
    st.caption("Generate a professional medical report with diagnosis, staging, and recommendations.")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Patient Name", "Ahmed Ali")
        age  = st.number_input("Age", 1, 120, 55)
        sex  = st.selectbox("Sex", ["male","female"])
        cr   = st.number_input("Creatinine (mg/dL)", 0.1, 20.0, 1.5, 0.1)
    with col2:
        acr  = st.number_input("ACR (mg/g)", 0.0, 5000.0, 30.0, 5.0)
        urea = st.number_input("Blood Urea (mg/dL)", 1.0, 300.0, 25.0)
        hemo = st.number_input("Hemoglobin (g/dL)", 1.0, 25.0, 13.0, 0.5)

    if st.button("📄 Generate PDF Report", use_container_width=True, type="primary"):
        payload = {"patient": {"name": name, "age": age, "sex": sex},
                   "lab_values": {"creatinine": cr, "acr": acr, "blood_urea": urea, "hemoglobin": hemo}}
        with st.spinner("Generating PDF..."):
            data, err = api_post("/report", payload)
        if err:
            st.error(f"Error: {err}")
        else:
            filename = data.get("filename","report.pdf")
            download_url = f"{API_BASE}/report/download/{filename}"
            st.success(f"✅ Report generated: `{filename}`")
            st.markdown(f"### [⬇️ Download PDF Report]({download_url})")
            st.info("Click the link above to download your PDF report.")
