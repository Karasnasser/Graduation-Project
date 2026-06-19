import React from "react";
import { FileText, ExternalLink } from "lucide-react";
import { motion } from "framer-motion";
import Footer from "../components/Footer";

const papers = [
  {
    id: "1",
    title:
      "A predictive model for progression of chronic kidney disease to kidney failure",
    source: "JAMA",
    date: "April 21, 2011",
    link: "https://github.com/settings/profile",
  },
  {
    id: "2",
    title: "A predictive model for progression of CKD",
    source: "Medicine (Baltimore)",
    date: "July 20, 2021",
    link: "https://github.com/settings/profile",
  },
  {
    id: "3",
    title:
      "Multinational Assessment of Accuracy of Equations for Predicting Risk of Kidney Failure",
    source: "JAMA",
    date: "April 21, 2011",
    link: "https://github.com/settings/profile",
  },
  {
    id: "4",
    title:
      "A Predictive Model for Progression of Chronic Kidney Disease to Kidney Failure",
    source: "JAMA Network",
    date: "April 20, 2011",
    link: "https://github.com/settings/profile",
  },
];

const References = () => {
  return (
    <>
      <div className="min-h-screen bg-linear-to-b from-blue-50 to-white py-10 px-4">
        <div className="max-w-5xl mx-auto">

          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-10"
          >
            <h1 className="text-3xl font-bold text-blue-900">
              Medical Research References
            </h1>
            <p className="text-gray-600 mt-2">
              Trusted clinical studies and kidney disease prediction models
            </p>
          </motion.div>

          {/* Cards */}
          <div className="grid gap-6 md:grid-cols-2">
            {papers.map((paper, index) => (
              <motion.div
                key={paper.id}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.02 }}
                className="bg-white border border-blue-100 shadow-md rounded-2xl p-5 hover:shadow-xl transition"
              >
                <div className="flex items-start gap-3">

                  {/* icon */}
                  <div className="bg-blue-100 p-2 rounded-lg">
                    <FileText className="text-blue-700" />
                  </div>

                  {/* content */}
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-800">
                      {paper.title}
                    </h3>

                    <p className="text-sm text-gray-500 mt-1">
                      {paper.source} • {paper.date}
                    </p>

                    {/* link */}
                    <a
                      href={paper.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 mt-3 text-blue-700 hover:text-blue-900 text-sm font-medium"
                    >
                      <ExternalLink size={16} />
                      View Study
                    </a>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
      <Footer />
    </>
  );
};

export default References;