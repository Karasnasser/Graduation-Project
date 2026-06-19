import React from "react";

export default function RiskAssessmentSection() {
  return (
    <section className="bg-linear-to-br from-[#274690] to-[#3A5BA9]">
      <div className="max-w-7xl mx-auto px-6 py-10">
        <div className="grid lg:grid-cols-2 gap-16 items-center">

          {/* Left Side */}
          <div>
            <p className="uppercase tracking-[4px] text-cyan-300 font-semibold text-sm mb-4">
              WHAT IS KIDNEFY?
            </p>

            <h3 className="text-5xl font-bold text-white leading-tight mb-8">
              AI-powered kidney disease prediction and Risk Analysis
            </h3>

            <p className="text-blue-100 text-lg leading-relaxed mb-6">
              Kidnefy helps patients detect chronic kidney disease risks using advanced artificial intelligence and laboratory data analysis.
            </p>

            <p className="text-blue-100 text-lg leading-relaxed mb-8">
              Our platform evaluates kidney function indicators,
              predicts potential risks, and generates comprehensive
              reports to support informed medical decisions.
            </p>

            <div className="space-y-5">
              {[
                "Early detection of kidney disease risks",
                "AI-driven prediction with high accuracy",
                "Instant interpretation of laboratory results",
                "Downloadable medical reports and recommendations",
              ].map((item, index) => (
                <div key={index} className="flex items-center gap-3">
                  <div className="w-6 h-6 rounded-full border-2 border-cyan-300 flex items-center justify-center text-cyan-300">
                    ✓
                  </div>
                  <p className="text-white">{item}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Right Side */}
          <div className="bg-white rounded-4xl p-10 shadow-lg">
            <div className="flex justify-between items-center mb-8">
              <span className="font-semibold text-[#3569db] text-sm">
                Sample Prediction
              </span>

              <span className="bg-red-100 text-red-700 px-4 py-2 rounded-full text-sm font-semibold">
                LOW RISK
              </span>
            </div>

            <div className="flex items-end gap-3 mb-6">
              <h3 className="text-7xl font-bold text-[#274690]">
                25%
              </h3>

              <span className="text-gray-500 pb-3">
                Risk Score
              </span>
            </div>

            <div className="w-full h-3 bg-gray-200 rounded-full mb-10">
              <div className="w-[22%] h-full bg-linear-to-r from-[#3569db] to-[#3b82f6] rounded-full" />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-[#f5f7fc] p-5 rounded-2xl">
                <h4 className="font-semibold text-[#274690]">
                  Serum Creatinine
                </h4>
              </div>

              <div className="bg-[#f5f7fc] p-5 rounded-2xl">
                <h4 className="font-semibold text-[#274690]">
                  eGFR Analysis
                </h4>
              </div>

              <div className="bg-[#f5f7fc] p-5 rounded-2xl">
                <h4 className="font-semibold text-[#274690]">
                  HbA1c
                </h4>
              </div>

              <div className="bg-[#f5f7fc] p-5 rounded-2xl">
                <h4 className="font-semibold text-[#274690]">
                  UACR
                </h4>
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}