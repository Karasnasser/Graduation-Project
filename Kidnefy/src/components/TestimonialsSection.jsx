import React from "react";
import { FaQuoteLeft } from "react-icons/fa";
import { motion } from "framer-motion";

const testimonials = [
  {
    name: "Dr. James Wilson",
    role: "Nephrologist, General Hospital",
    text:
      "NephroAI has revolutionized how we screen for early-stage CKD. The accuracy is unmatched and it saves us valuable time in diagnosis.",
  },
  {
    name: "Dr. Emily Chen",
    role: "Internal Medicine Specialist",
    text:
      "The interface is incredibly user-friendly. I was able to upload my patient’s data and get a second opinion within minutes.",
  },
  {
    name: "Michael Brown",
    role: "Patient",
    text:
      "As a patient, knowing that AI is checking my health gives me peace of mind. The report was easy to understand and very detailed.",
  },
];

export default function TestimonialsSection() {
  return (
    <section className="bg-linear-to-br from-[#f5f7fc] to-[#eef3ff] py-24 px-6 relative overflow-hidden">

      {/* Glow Background */}
      <div className="absolute w-[500px] h-[500px] bg-blue-300 blur-3xl opacity-20 rounded-full top-[-150px] left[-150px]" />
      <div className="absolute w-[400px] h-[400px] bg-indigo-300 blur-3xl opacity-20 rounded-full bottom-[-150px] right-[-150px]" />

      {/* Title */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        viewport={{ once: true }}
        className="text-center mb-16"
      >
        <h2 className="text-3xl md:text-4xl font-semibold text-[#274690] mb-3">
          Trusted by Professionals
        </h2>

        <p className="text-gray-500">
          See what medical experts and patients are saying about NephroAI.
        </p>
      </motion.div>

      {/* Cards */}
      <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto relative z-10">

        {testimonials.map((item, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.15 }}
            viewport={{ once: true }}
            whileHover={{ scale: 1.05 }}
            className="bg-white/80 backdrop-blur-lg p-8 rounded-3xl shadow-lg border border-white relative text-center"
          >

            {/* Quote Icon */}
            <div className="absolute -top-5 left-1/2 transform -translate-x-1/2 bg-[#274690] text-white p-3 rounded-full shadow-md">
              <FaQuoteLeft />
            </div>

            {/* Text */}
            <p className="text-gray-600 mt-6 text-sm leading-relaxed">
              "{item.text}"
            </p>

            {/* Name */}
            <h4 className="mt-6 font-semibold text-gray-800">
              {item.name}
            </h4>

            {/* Role */}
            <p className="text-gray-400 text-sm">
              {item.role}
            </p>

          </motion.div>
        ))}

      </div>

    </section>
  );
}