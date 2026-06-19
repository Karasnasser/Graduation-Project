import React from "react";
import CountUp from "react-countup";
import { motion } from "framer-motion";
import {
  FaUser,
  FaClipboardCheck,
  FaRobot,
  FaFileMedicalAlt,
  FaHeartbeat,
} from "react-icons/fa";

const stats = [
  {
    icon: <FaUser />,
    value: 12000,
    suffix: "+",
    title: "Patients Analyzed",
  },
  {
    icon: <FaClipboardCheck />,
    value: 85000,
    suffix: "+",
    title: "Tests Completed",
  },
  {
    icon: <FaRobot />,
    value: 99,
    suffix: "%",
    title: "AI Accuracy",
  },
  {
    icon: <FaFileMedicalAlt />,
    value: 50000,
    suffix: "+",
    title: "Reports Generated",
  },
  {
    icon: <FaHeartbeat />,
    value: 99,
    suffix: "%",
    title: "Prediction Accuracy",
  },
];

export default function StatsSection() {
  return (
    <section className="relative py-12 px-6 bg-[#f5f7fc] overflow-hidden">
      <div className="absolute w-500px h-500px bg-blue-300 blur-3xl opacity-20 rounded-full -top-40 -left-40" />
      <div className="absolute w-400px h-400px bg-cyan-300 blur-3xl opacity-20 rounded-full -bottom-40 -right-40" />

      <div className="max-w-7xl mx-auto relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <h2 className="text-xl md:text-4xl font-semibold text-blue-900 mb-4">
            Kidnefy in Numbers
          </h2>

          <p className="text-gray-500 max-w-2xl mx-auto text-lg">
            Trusted by thousands of users for early kidney disease detection
            powered by advanced Artificial Intelligence.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-6">
          {stats.map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{
                duration: 0.5,
                delay: index * 0.1,
              }}
              whileHover={{
                scale: 1.05,
                y: -5,
              }}
              className="bg-white border border-gray-100 rounded-3xl p-8 text-center hover:bg-linear-to-br hover:from-[#274690] hover:to-[#3A5BA9] shadow-xl transition-all duration-300 group"
            >
              <div className="text-5xl text-[#3569db] group-hover:text-white flex justify-center mb-5 transition-colors duration-300">
                {item.icon}
              </div>

              <h3 className="text-4xl font-bold text-[#274690] group-hover:text-white mb-3 transition-colors duration-300">
                <CountUp
                  end={item.value}
                  duration={3}
                  enableScrollSpy
                  scrollSpyOnce
                  decimals={item.value % 1 !== 0 ? 1 : 0}
                />
                {item.suffix}
              </h3>

              <p className="text-gray-600 group-hover:text-white font-medium text-lg transition-colors duration-300">
                {item.title}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}