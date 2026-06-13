import React from "react";
import { FaBrain, FaClock, FaShieldAlt } from "react-icons/fa";
import { motion } from "framer-motion";

const features = [
  {
    icon: <FaBrain />,
    title: "AI-Powered Analysis",
    desc: "Our deep learning models analyze complex renal biomarkers instantly, identifying subtle patterns that traditional methods might miss.",
  },

  {
    icon: <FaShieldAlt />,
    title: "Smart CKD Detection",
    desc: "We combine advanced machine learning algorithms with medical expertise to provide reliable, fast, and accessible kidney health monitoring.",
  },
  
  {
    icon: <FaClock />,
    title: "Real-Time Results",
    desc: "Get comprehensive diagnostic reports in seconds, not days. Fast-track your treatment plans with immediate insights.",
  },
];

export default function FeaturesSection() {
  return (
    <section className="py-12 bg-white px-6 md:px-10">
      <div className="max-w-6xl mx-auto">
        {/* Title */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl md:text-4xl font-semibold text-blue-900 mb-3">
            Why Choose Kidnefy?
          </h2>

          {/* <p className="text-gray-500 max-w-2xl mx-auto text-sm md:text-base leading-relaxed">
            Intelligent kidney care with fast, accurate insights designed to keep your renal health on track.
          </p> */}
        </motion.div>

        {/* Cards */}
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
              whileHover={{ scale: 1.03 }}
              className="bg-gray-50 p-8 rounded-3xl shadow-sm hover:shadow-lg transition duration-300 flex flex-col justify-between min-h-80 w-full"
            >
              {/* Icon */}
              <div className="w-14 h-14 flex items-center justify-center bg-blue-100 text-blue-900 rounded-2xl mb-6 text-2xl">
                {feature.icon}
              </div>

              <div className="flex-1">
                {/* Title */}
                <h3 className="font-bold text-gray-800 mb-3 text-xl">
                  {feature.title}
                </h3>

                {/* Description */}
                <p className="text-gray-600 text-sm md:text-base leading-relaxed">
                  {feature.desc}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}