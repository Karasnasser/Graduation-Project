import React, { useState, useEffect } from "react";
import Footer from "../components/Footer";
// افتراضياً عندنا user من login


export default function SmartAlerts() {
  const [text, setText] = useState("");
  const [patientId, setPatientId] = useState(""); // ID يظهر تلقائي
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const savedPatientId =
      localStorage.getItem("patientId") || "demo_patient_001";

    setPatientId(savedPatientId);
  }, []);

  const analyze = () => {
    if (!text.trim()) return;

    setLoading(true);

    setTimeout(() => {
      setResult({
        level: "EMERGENCY",
        conditions: "Possible kidney complication",
        recommendation: "See a doctor immediately for diagnosis and treatment",
        patient: patientId
      });
      setLoading(false);
    }, 1200);
  };

  return (
    <>
    <section className="min-h-screen bg-linear-to-b from-blue-50 to-white p-6 md:p-10">
      {/* HEADER */}
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold text-blue-900">
          Risk Detector
        </h1>
        <p className="text-gray-600 mt-2">
          AI-powered anomaly detection for real-time patient monitoring
        </p>
      </div>

      {/* MAIN GRID */}
      {/* MAIN CARD (Merged) */}
      <div className="
        bg-white/70 backdrop-blur-xl
        border border-white/40
        shadow-2xl
        p-8 rounded-3xl
        max-w-4xl mx-auto
        space-y-5
        hover:scale-[1.01]
        transition
      ">
        {/* HEADER */}
        <div>
          <h2 className="text-2xl font-bold text-[#274690] mb-2">
            Risk Detector Dashboard
          </h2>

          <p className="text-gray-600 leading-relaxed">
            Run anomaly detection to identify abnormal kidney patterns early.
          </p>
        </div>

        {/* SCAN BUTTON */}
        <button className="
          w-full
          bg-linear-to-r from-[#274690] to-blue-500
          hover:from-[#1e3a8a] hover:to-blue-600
          text-white font-semibold
          px-6 py-3 rounded-2xl
          shadow-lg
          transition
        ">
          Scan for Anomalies
        </button>
      </div>

    </section>
      <Footer />
      </>
  );
}