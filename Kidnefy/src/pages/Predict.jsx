import React, { useState } from "react";
import { RadialBarChart, RadialBar } from "recharts";
import jsPDF from "jspdf";
import Footer from "../components/Footer";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

export default function PredictPage() {

  const [form, setForm] = useState({
    name: "",
    age: "",
    gender: "",

    creatinine: "",
    blood_urea: "",
    egfr: "",
    urine_albumin: "",

    sodium: "",
    potassium: "",
    calcium: "",
    phosphorus: "",
    hemoglobin: ""
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // NLP Symptom Analysis state
  const [symptomsText, setSymptomsText] = useState("");
  const [symptomsResult, setSymptomsResult] = useState(null);
  const [symptomsLoading, setSymptomsLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value
    });
  };

  // ======================
  // AI Prediction
  // ======================

  const predict = async () => {

    setLoading(true);

    await new Promise((res) => setTimeout(res, 1500));

    const creatinine = parseFloat(form.creatinine || 1);
    const urea = parseFloat(form.blood_urea || 20);
    const hb = parseFloat(form.hemoglobin || 13);
    const egfr = parseFloat(form.egfr || 90);
    const albumin = parseFloat(form.urine_albumin || 0);

    let risk = Math.min(
      100,
      Math.round(
        creatinine * 20 +
        urea * 0.4 +
        (13 - hb) * 4 +
        albumin * 0.3 +
        (90 - egfr) * 0.5
      )
    );

    let stage = "G1";

    if (risk > 80) stage = "G5";
    else if (risk > 65) stage = "G4";
    else if (risk > 50) stage = "G3";
    else if (risk > 30) stage = "G2";

    let level = "Low";

    if (risk > 70) level = "High";
    else if (risk > 40) level = "Moderate";

    const recommendations = {
      Low: "Kidney function appears normal. Maintain a healthy lifestyle.",
      Moderate: "Monitor kidney function regularly and consult a doctor.",
      High: "High risk detected. Immediate nephrologist consultation recommended."
    };

    setResult({
      risk,
      stage,
      level,
      recommendations: recommendations[level]
    });

    setLoading(false);
  };

  // ======================
  // NLP Symptom Analysis
  // ======================

  const analyzeSymptoms = async () => {
    if (!symptomsText.trim()) return;

    setSymptomsLoading(true);

    await new Promise((res) => setTimeout(res, 1500));

    const textLower = symptomsText.toLowerCase();

    // Simple NLP keyword detection
    const keywords = {
      High: ["swelling", "swollen", "blood", "severe", "pain", "fever", "infection", "urine", "vomit", "unconscious"],
      Moderate: ["fatigue", "tired", "nausea", "dizziness", "weight loss", "loss of appetite", "itching", "cramps"],
      Low: ["mild", "occasional", "slight"]
    };

    let level = "Low";
    let matchCounts = { High: 0, Moderate: 0, Low: 0 };

    for (const [lvl, words] of Object.entries(keywords)) {
      for (const word of words) {
        if (textLower.includes(word)) {
          matchCounts[lvl]++;
        }
      }
    }

    if (matchCounts.High > 0) level = "High";
    else if (matchCounts.Moderate > 0) level = "Moderate";

    const conditions = {
      High: "Possible kidney infection or acute kidney injury — urgent evaluation needed.",
      Moderate: "Possible early-stage kidney dysfunction — consider further testing.",
      Low: "No significant kidney-related symptoms detected."
    };

    const recommendations = {
      High: "Immediate medical attention recommended. Visit a nephrologist or emergency room.",
      Moderate: "Schedule an appointment with a primary care physician for kidney function tests.",
      Low: "Maintain hydration and a balanced diet. Monitor for any changes."
    };

    setSymptomsResult({
      level,
      condition: conditions[level],
      recommendation: recommendations[level]
    });

    setSymptomsLoading(false);
  };

  // ======================
  // PDF Export
  // ======================

 const exportPDF = () => {
  if (!result) return;

  const doc = new jsPDF();

  const egfr = Number(form.egfr);

  const stage =
    egfr < 15 ? 5 :
    egfr < 30 ? 4 :
    egfr < 60 ? 3 :
    egfr < 90 ? 2 : 1;

  const riskLevel =
    egfr < 15 ? "Critical" :
    egfr < 30 ? "High Risk" :
    egfr < 60 ? "Moderate Risk" : "Low Risk";

  // ================= HEADER =================
  doc.setFillColor(39, 70, 144);
  doc.rect(0, 0, 210, 25, "F");

  doc.setTextColor(255, 255, 255);
  doc.setFontSize(16);
  doc.text("Kidnefy Report", 20, 16);

  // ================= PATIENT CARD =================
  doc.setTextColor(0, 0, 0);
  doc.setFillColor(255, 255, 255);
  doc.roundedRect(20, 35, 170, 25, 3, 3, "F");

  doc.setFontSize(11);
  doc.text(`Name: ${form.name}`, 25, 45);
  doc.text(`Age: ${form.age}`, 80, 45);
  doc.text(`Gender: ${form.gender}`, 130, 45);
  doc.text(`Date: ${new Date().toLocaleDateString()}`, 25, 55);

  // ================= RESULT CARDS =================
  // Stage
  doc.roundedRect(20, 70, 80, 35, 3, 3);
  doc.text("CKD Stage", 25, 80);
  doc.setFontSize(18);
  doc.text(`Stage ${stage}`, 25, 92);

  // Risk
  doc.roundedRect(110, 70, 80, 35, 3, 3);
  doc.setFontSize(11);
  doc.text("Risk Level", 115, 80);
  doc.setFontSize(16);
  doc.text(riskLevel, 115, 92);

  // Progress bar
  doc.setFillColor(220, 220, 220);
  doc.rect(115, 97, 60, 4, "F");

  doc.setFillColor(39, 70, 144);
  doc.rect(115, 97, (result.risk / 100) * 60, 4, "F");

  // ================= INDICATORS =================
  doc.setFontSize(11);
  doc.text("Probability: " + result.risk + "%", 20, 115);
  doc.text("eGFR: " + form.egfr, 80, 115);
  doc.text("Creatinine: " + form.creatinine, 130, 115);

  // ================= LAB RESULTS =================
  doc.roundedRect(20, 125, 170, 60, 3, 3);
  doc.text("Lab Results", 25, 135);

  let y = 145;

  const labData = [
    ["Blood Urea", form.blood_urea],
    ["Urine Albumin", form.urine_albumin],
    ["Sodium", form.sodium],
    ["Potassium", form.potassium],
    ["Calcium", form.calcium],
    ["Phosphorus", form.phosphorus],
    ["Hemoglobin", form.hemoglobin],
  ];

  labData.forEach(([key, val], i) => {
    const x = i % 2 === 0 ? 25 : 110;
    if (i % 2 === 0 && i !== 0) y += 7;

    doc.text(`${key}: ${val}`, x, y);
  });

  // ================= RECOMMENDATIONS =================
  doc.roundedRect(20, 195, 170, 50, 3, 3);
  doc.text("Recommendations", 25, 205);

  const splitText = doc.splitTextToSize(result.recommendations, 160);
  doc.text(splitText, 25, 215);

  // ================= FOOTER =================
  doc.setFontSize(9);
  doc.setTextColor(150);

  doc.text(
    "This report is generated by AI and should not replace medical advice.",
    20,
    285
  );

  doc.text(new Date().toLocaleDateString(), 170, 285);

  doc.save("KidneyCare-Report.pdf");
};

  return (
    <>
    
{/* ================= FORM ================= */}

{!result && (

<section className="min-h-screen bg-[#f5f7fc] p-10">

<div className="max-w-6xl mx-auto">

{/* HEADER */}
{/* <div className="text-center mb-14">
  <h1 className="text-5xl md:text-6xl font-extrabold text-[#274690] tracking-tight">
    KidneyCare
  </h1>

  <p className="text-gray-500 mt-4 text-lg md:text-xl">
    AI-Powered Kidney Disease Prediction System
  </p>
</div> */}

<div className="flex flex-col gap-10">

{/* LEFT SIDE */}
<div className="flex flex-col gap-8">

{/* Patient Info */}
<div className="bg-white backdrop-blur-xl border border-white/40 shadow-2xl p-8 rounded-3xl">

<h2 className="text-2xl font-bold mb-6 text-[#274690]">
Patient Information
</h2>

<div className="grid grid-cols-1 md:grid-cols-2 gap-4">

<input
name="name"
placeholder="Patient Name"
onChange={handleChange}
className="
w-full 
border 
border-gray-200 
rounded-xl 
p-4 
focus:outline-none 
focus:ring-2 
focus:ring-blue-500 
focus:border-transparent 
transition
bg-white/80
outline-none
shadow-sm
"
/>

<input
name="email"
type="email"
placeholder="Email"
onChange={handleChange}
className="
w-full 
border 
border-gray-200 
rounded-xl 
p-4 
focus:outline-none 
focus:ring-2 
focus:ring-blue-500 
focus:border-transparent 
transition
bg-white/80
outline-none
shadow-sm
"
/>

<div className="flex flex-col gap-2">
<label htmlFor="age" className="text-sm font-medium text-gray-500">
Date of Birth
</label>
<input
id="age"
name="age"
type="date"
onChange={handleChange}
className="w-full 
border 
border-gray-200 
rounded-xl 
p-4 
focus:outline-none 
focus:ring-2 
focus:ring-blue-500 
focus:border-transparent 
transition
bg-white/80
outline-none
shadow-sm"
/>
</div>

<div className="flex flex-col gap-2">
<label htmlFor="gender" className="text-sm font-medium text-gray-500">
Gender
</label>
<div className="relative">
<select
id="gender"
name="gender"
onChange={handleChange}
className="appearance-none w-full 
border 
border-gray-200 
rounded-xl 
p-4 
focus:outline-none 
focus:ring-2 
focus:ring-blue-500 
focus:border-transparent 
transition
bg-white/80
outline-none
shadow-sm"
>
<option value="" className="text-gray-400">Select Gender</option>
<option value="male">Male</option>
<option value="female">Female</option>
</select>
<span className="pointer-events-none absolute inset-y-0 right-4 flex items-center text-gray-500">
<svg width="16" height="16" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M6 8L10 12L14 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
</svg>
</span>
</div>
</div> 

</div>
</div>

{/* Upload */}
{/* <div className="bg-white/70 backdrop-blur-xl border border-white/40 shadow-2xl p-8 rounded-3xl">

<h2 className="text-2xl font-bold mb-6 text-[#274690]">
Upload Lab Test
</h2>

<label className="border-2 border-dashed border-blue-300 rounded-3xl p-10 flex flex-col items-center justify-center cursor-pointer hover:bg-blue-50 transition">

<p className="text-gray-600">
Upload PDF or Image
</p>

<input
type="file"
accept=".pdf,image/*"
className="hidden"
onChange={(e) => {
const file = e.target.files[0];
if (!file) return;

alert(`"${file.name}" uploaded successfully`);
}}
/>

</label>

</div> */}

</div>

{/* RIGHT SIDE */}
<div className="flex flex-col gap-8">

{/* Kidney Function */}
<div className="bg-white backdrop-blur-xl border border-white/40 shadow-2xl p-8 rounded-3xl">

<h2 className="text-2xl font-bold mb-6 text-[#274690]">
Kidney Functions
</h2>

<div className="grid md:grid-cols-2 gap-4">

{[
["creatinine","Serum Creatinine"],
["blood_urea","Blood Urea"],
["egfr","eGFR"],
["urine_albumin","Urine Albumin"]
].map(([name, placeholder]) => (

<input
key={name}
name={name}
placeholder={placeholder}
onChange={handleChange}
className="w-full 
border 
border-gray-200 
rounded-xl 
p-4 
focus:outline-none 
focus:ring-2 
focus:ring-blue-500 
focus:border-transparent 
transition
bg-white/80
outline-none
shadow-sm"
/>

))}

</div>
</div>

{/* Electrolytes */}
<div className="bg-white backdrop-blur-xl border border-white/40 shadow-2xl p-8 rounded-3xl">

<h2 className="text-2xl font-bold mb-6 text-[#274690]">
Electrolytes & Metabolic
</h2>

<div className="grid md:grid-cols-2 gap-4">

{[
["sodium","Sodium"],
["potassium","Potassium"],
["calcium","Calcium"],
["phosphorus","Phosphorus"],
["hemoglobin","Hemoglobin"]
].map(([name, placeholder]) => (

<input
key={name}
name={name}
placeholder={placeholder}
onChange={handleChange}
className="w-full 
border 
border-gray-200 
rounded-xl 
p-4 
focus:outline-none 
focus:ring-2 
focus:ring-blue-500 
focus:border-transparent 
transition
bg-white/80
outline-none
shadow-sm"
/>

))}

</div>
</div>

{/* NLP Symptom Analysis */}
<div className="bg-white backdrop-blur-xl border border-white/40 shadow-2xl p-8 rounded-3xl">

<h2 className="text-2xl font-bold mb-6 text-[#274690]">
Symptom Analysis
</h2>
<div className="flex flex-col gap-4">
<label htmlFor="gender" className="text-sm font-medium text-gray-500">
If you would like to enter symptoms (in Arabic or English), the artificial intelligence will classify the condition immediately
</label>


<div className="relative">
<textarea
  placeholder="Example: I feel swelling in my feet and extreme fatigue."
  value={symptomsText}
  onChange={(e) => setSymptomsText(e.target.value)}
  className="
    w-full 
    border 
  border-gray-200 
    rounded-xl 
    p-4 
    focus:outline-none 
    focus:ring-2 
    focus:ring-blue-500 
    focus:border-transparent 
    transition
  bg-white/80
    outline-none
    shadow-sm
  "
/>

</div>
</div>
</div>

{/* <div className="grid grid-cols-2 gap-4 mt-8"> */}
<div className="col-span-2 flex justify-end w-full">
  <button
    onClick={predict}
    className="bg-linear-to-r from-[#2747a5] via-[#3569db] to-[#3b82f6] text-white px-8 py-4 rounded-2xl font-semibold shadow-lg transition-all duration-300"
  >
    {loading ? "Analyzing..." : "Run AI Diagnosis →"}
  </button>


  {/* <button
  onClick={() => navigate("/smart-alerts")}
  className="bg-linear-to-r from-green-500 to-emerald-600 hover:scale-105 transition text-white py-4 rounded-2xl font-semibold shadow-lg"
  >
  Smart Alerts →
  </button> */}

</div>
</div>

</div>
</div>
</section>




)}

{/* ================= RESULT DASHBOARD ================= */}

{result && (

<section className="min-h-screen bg-[#f5f7fc] p-10">

<div className="max-w-6xl mx-auto space-y-6">

<button
  onClick={() => {
    setResult(null);
    setForm({
      name: "",
      age: "",
      gender: "",
      creatinine: "",
      blood_urea: "",
      egfr: "",
      urine_albumin: "",
      sodium: "",
      potassium: "",
      calcium: "",
      phosphorus: "",
      hemoglobin: ""
    });
  }}
  className="bg-linear-to-r from-[#2747a5] via-[#3569db] to-[#3b82f6] text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
>
  ← Back
</button>

{/*  Analysis Complete */}
<div className="bg-linear-to-r from-green-50 to-emerald-50 text-green-800 p-5 rounded-xl flex justify-between items-center shadow-md hover:shadow-lg transition-shadow">
  <p className="font-bold text-lg">Analysis Complete</p>
  <span className="text-2xl">✅</span>
</div>

{/* ================= PATIENT INFO ================= */}
<div className="bg-white p-6 rounded-2xl shadow-sm flex justify-between flex-wrap gap-4">

<div>
  <p className="text-gray-500 text-sm">Name</p>
  <p className="font-bold">{form.name || "—"}</p>
</div>

<div>
  <p className="text-gray-500 text-sm">Age</p>
  <p className="font-bold">{form.age || "—"}</p>
</div>

<div>
  <p className="text-gray-500 text-sm">Gender</p>
  <p className="font-bold">{form.gender}</p>
</div>

<div>
  <p className="text-gray-500 text-sm">Date</p>
  <p className="font-bold">
    {new Date().toLocaleDateString()}
  </p>
</div>

</div>

{/* ================= LOGIC ================= */}
{(() => {
  const egfr = Number(form.egfr);
  const creatinine = Number(form.creatinine);
  const hemoglobin = Number(form.hemoglobin);

  const stage =
    egfr < 15 ? 5 :
    egfr < 30 ? 4 :
    egfr < 60 ? 3 :
    egfr < 90 ? 2 : 1;

  const stageText =
    egfr < 15 ? "Kidney Failure" :
    egfr < 30 ? "Severe decrease" :
    egfr < 60 ? "Moderate decrease" :
    egfr < 90 ? "Mild decrease" : "Normal";

  const riskLevel =
    egfr < 15 ? "Critical" :
    egfr < 30 ? "High Risk" :
    egfr < 60 ? "Moderate Risk" : "Low Risk";

  return (

  <>
  {/* ================= RESULT ================= */}
  <div className="grid md:grid-cols-2 gap-6">

  {/* Stage */}
  <div className="bg-white p-8 rounded-2xl shadow-sm text-center">
    <p className="text-gray-500">CKD Stage</p>
    <h2 className="text-3xl font-bold text-blue-700 mt-2">
      Stage {stage}
    </h2>
    <p className="text-gray-400 mt-2">{stageText}</p>
  </div>

  {/* Risk */}
  <div className="bg-green-100 p-8 rounded-2xl shadow-sm">
    <p className="text-gray-600">Risk Level</p>

    <h2 className={`text-2xl font-bold mt-2 ${
      riskLevel === "Critical"
        ? "text-red-600"
        : "text-green-800"
    }`}>
      {riskLevel}
    </h2>

    <div className="w-full bg-green-200 h-2 rounded-full mt-4">
      <div
        className="bg-green-700 h-2 rounded-full"
        style={{ width: `${result.risk}%` }}
      ></div>
    </div>
  </div>
  
  {/* probability */}
  {/* <div className="bg-white p-8 rounded-2xl shadow-sm w-full">
    <p className="text-3xl font-bold text-green-700 mt-2 text-center">Probability</p>
    <p className="text-gray-600 font-bold text-2xl mt-4 text-center">
      {result.risk}%
    </p>
    <p className="text-gray-400 mt-2 text-center">
      {result.risk < 50 ? "Low" : result.risk < 80 ? "Moderate" : "High"} risk of developing CKD
    </p>
  </div> */}

  </div>
      

  {/* ================= INDICATORS ================= */}
  {/* <div className="bg-white p-6 rounded-2xl shadow-sm flex justify-between flex-wrap gap-4">

  </div> */}

  {/* ================= LAB RESULTS ================= */}
  <div className="bg-white p-6 rounded-2xl shadow-sm">

  <h2 className="font-bold mb-4 text-gray-800">
    Lab Results
  </h2>

  <div className="grid md:grid-cols-3 gap-4 text-sm text-gray-600">

  <p className="font-bold">eGFR: {form.egfr}</p>
  <p className="font-bold">Creatinine: {form.creatinine}</p>
  <p className="font-bold">Blood Urea: {form.blood_urea}</p>
  <p className="font-bold">Urine Albumin: {form.urine_albumin}</p>
  <p className="font-bold">Sodium: {form.sodium}</p>
  <p className="font-bold">Potassium: {form.potassium}</p>
  <p className="font-bold">Calcium: {form.calcium}</p>
  <p className="font-bold">Phosphorus: {form.phosphorus}</p>
  <p className="font-bold">Hemoglobin: {form.hemoglobin}</p>

  </div>

  </div>

  {/* ================= AI EXPLANATION ================= */}
  <div className="bg-white p-6 rounded-2xl shadow-sm">

  <h3 className="font-bold mb-3 text-gray-800">
    AI Explanation
  </h3>

  <ul className="list-disc pl-5 text-gray-600 space-y-1">

  {creatinine > 5 && (
    <li>Creatinine is very high</li>
  )}

  {egfr < 15 && (
    <li>Severe kidney failure - urgent care needed</li>
  )}

  {hemoglobin < 10 && (
    <li>Possible anemia detected</li>
  )}

  {egfr >= 60 && (
    <li>Kidney function is relatively normal</li>
  )}

  </ul>

  </div>

  </>
  );
})()}

{/* ================= BUTTONS ================= */}
<div className="grid md:grid-cols-2 gap-5">

  {/* Download PDF Button */}
  <motion.button
    whileHover={{ scale: 1.04, y: -3 }}
    whileTap={{ scale: 0.96 }}
    onClick={exportPDF}
    className="
      relative
      overflow-hidden
      bg-linear-to-r
      from-[#2747a5] 
      via-[#3569db] 
      to-[#3b82f6]
      text-white
      font-bold
      px-6
      py-3
      rounded-xl
      shadow-[0_8px_20px_rgba(59,130,246,0.3)]
      transition-all
      duration-300
    "
  >
    <span className="relative z-10 text-sm">
      Download PDF 
    </span>

    <motion.span
      initial={{ x: "-120%" }}
      whileHover={{ x: "220%" }}
      transition={{ duration: 0.8 }}
      className="
        absolute
        top-0
        left-0
        w-1/3
        h-full
        bg-white/20
        skew-x-12
      "
    />
  </motion.button>

{/* Risk Detector button */}
<button
onClick={() => navigate("/smart-alerts")}
className="bg-linear-to-r from-green-500 to-emerald-600 hover:scale-105 transition text-white py-4 rounded-2xl font-semibold shadow-lg"
>
Risk Detector →
</button>
  
  {/* New Test */}
  {/* <motion.button
    whileHover={{ scale: 1.04, y: -3 }}
    whileTap={{ scale: 0.96 }}
    onClick={() => {
      setResult(null);
      setForm({
        name: "",
        age: "",
        gender: "male",
        creatinine: "",
        blood_urea: "",
        egfr: "",
        urine_albumin: "",
        sodium: "",
        potassium: "",
        calcium: "",
        phosphorus: "",
        hemoglobin: ""
      });
    }}
    className="
      relative
      overflow-hidden
      bg-linear-to-r
      from-green-500
      to-emerald-600
      text-white
      font-bold
      px-6
      py-3
      rounded-xl
      shadow-[0_8px_20px_rgba(16,185,129,0.3)]
      transition-all
      duration-300
    "
  >
    <span className="relative z-10 text-sm">
      New Test 
    </span>

    <motion.span
      initial={{ x: "-120%" }}
      whileHover={{ x: "220%" }}
      transition={{ duration: 0.8 }}
      className="
        absolute
        top-0
        left-0
        w-1/3
        h-full
        bg-white/20
        skew-x-12
      "
    />
  </motion.button> */}

</div>

</div>
</section>

)}


<Footer />

</>
  );
}
