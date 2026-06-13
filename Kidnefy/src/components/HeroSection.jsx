import React from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import heroImg from "../assets/login-medical2.jpg";

export default function HeroSection() {
  const navigate = useNavigate();

  return (
    <section className="relative min-h-[70vh] w-full overflow-hidden">
      {/* Background Image */}
      <div className="absolute inset-0 z-0">
        <img
          src={heroImg}
          alt="Medical background"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-linear-to-r from-[#274690]/85 via-[#274690]/70 to-[#274690]/50"></div>
      </div>

      {/* Content Overlay */}
      <div className="relative z-10 flex items-center justify-center min-h-[65vh] px-4 md:px-8">
        <motion.div
          className="text-center max-w-2xl"
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
        >
          <motion.h1
            className="text-4xl md:text-6xl font-bold text-white leading-tight mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            Early Detection Saves
            <br />
            <span className="text-[#a8c8ff]">Your Kidneys</span>
          </motion.h1>

          <motion.p
            className="text-white/85 text-lg md:text-xl font-light mb-8 max-w-xl mx-auto leading-relaxed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2, ease: "easeOut" }}
          >
            Empowering nephrologists with AI-driven analytics for early-stage
            CKD detection with 99.9% precision accuracy. Start your proactive
            health journey today.
          </motion.p>

          <motion.div
            className="flex flex-col sm:flex-row justify-center gap-3 sm:gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3, ease: "easeOut" }}
          >
            <button
              onClick={() => {
                const user = localStorage.getItem("user");
                if (user) {
                  navigate("/predict");
                } else {
                  navigate("/SignIn");
                }
              }}
              className="bg-white text-[#274690] px-10 py-3 rounded-full font-bold text-base md:text-lg shadow-xl hover:shadow-2xl hover:scale-105 transition-all duration-300 active:scale-95"
            >
              Get Started
            </button>

            <button
              onClick={() => navigate("/about")}
              className="border-2 border-white/80 text-white px-10 py-3 rounded-full font-bold text-base md:text-lg hover:bg-white/15 hover:border-white transition-all duration-300 backdrop-blur-sm active:scale-95"
            >
              Learn More
            </button>
          </motion.div>
        </motion.div>
      </div>

      {/* Scroll Indicator */}
      <motion.div
        className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10"
        animate={{ y: [0, 10, 0] }}
        transition={{ repeat: Infinity, duration: 2.5 }}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-6 w-6 text-white/50"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M19 14l-7 7m0 0l-7-7m7 7V3"
          />
        </svg>
      </motion.div>
    </section>
  );
}
