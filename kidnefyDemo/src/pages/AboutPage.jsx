import React from "react";
import { motion } from "framer-motion";
import aboutImg from "../assets/login-medical.jpg";
import Footer from "../components/Footer";

export default function AboutPage() {
  return (
    <div className="relative min-h-screen flex flex-col overflow-hidden">

      {/* Background Image */}
      <div className="absolute inset-0">
        <img
          src={aboutImg}
          alt="bg"
          className="w-full h-full object-cover scale-105"
        />

        {/* Gradient Overlay */}
        <div className="absolute inset-0 bg-linear-to-r from-[#1e3a8acc] via-[#1e3a8acc] to-transparent"></div>
      </div>

      {/* MAIN CONTENT */}
      <div className="relative z-10 flex-1 flex items-center justify-center px-6 py-20">

        <div className="w-full max-w-6xl rounded-4xl bg-white/10 backdrop-blur-xl border border-white/20 shadow-2xl p-10 flex flex-col md:flex-row gap-10">

          {/* LEFT TEXT */}
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7 }}
            className="flex-1 text-white"
          >
            <h1 className="text-[38px] md:text-[50px] font-bold leading-tight mb-6">
              About Kidnefy:
              <br />
              Advanced AI for Kidney Disease Detection
            </h1>

            <p className="text-white/80 text-[17px] leading-relaxed max-w-lg">
              Our cutting-edge AI models enable early detection and continuous
              monitoring of kidney diseases. We provide accurate insights that
              support better medical decisions and improve patient outcomes.
            </p>
          </motion.div>

          {/* RIGHT SIDE */}
          <div className="flex-1 flex flex-col gap-6">
            
            {/* CARD 1 */}
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              whileHover={{ scale: 1.03 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="bg-white/90 backdrop-blur-lg rounded-2xl p-6 shadow-lg border border-white"
            >
              <h3 className="text-xl font-bold mb-3 text-gray-800">
                Kidnefy Features
              </h3>

              <p className="text-gray-600 text-[15px] leading-relaxed">
                - Early Stage Detection (stage 1-2)</p>
              <p className="text-gray-600 text-[15px] leading-relaxed">
                - Multi-Biomarkers Integration (kidney functions, Electrolytes)</p>
              <p className="text-gray-600 text-[15px] leading-relaxed">
                - Longitudinal Monitoring and Trend Analysis</p>
              <p className="text-gray-600 text-[15px] leading-relaxed">
                - Clinical Decision Support System</p>
            </motion.div>

            {/* CARD 2 */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}ش
              whileHover={{ scale: 1.03 }}
              transition={{ duration: 0.5 }}
              viewport={{ once: true }}
              className="relative bg-white/90 backdrop-blur-lg rounded-2xl p-6 shadow-lg border border-white"
            >
              <div className="absolute inset-0 rounded-2xl border border-blue-200 opacity-40 pointer-events-none"></div>

              <h3 className="text-xl font-bold mb-3 text-gray-800">
                Our Mission
              </h3>

              <p className="text-gray-600 text-[15px] leading-relaxed">
                Improving healthcare outcomes through the power of AI. We are
                dedicated to providing accurate, accessible, and early kidney
                disease detection to empower patients and doctors.
              </p>
            </motion.div>

            {/* CARD 3 */}
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              whileHover={{ scale: 1.03 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
              className="bg-white/90 backdrop-blur-lg rounded-2xl p-6 shadow-lg border border-white"
            >
              <h3 className="text-xl font-bold mb-3 text-gray-800">
                Our Team
              </h3>

              <p className="text-gray-600 text-[15px] leading-relaxed">
                A multidisciplinary team of nephrologists, data scientists, and
                AI engineers committed to medical innovation.
              </p>
            </motion.div>
            

          </div>

        </div>
      </div>

      {/* Footer */}
      <div className="relative z-10">
        <Footer />
      </div>
    </div>
  );
}