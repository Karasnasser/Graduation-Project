import React from "react";
import { FaLinkedin, FaTwitter, FaGithub } from "react-icons/fa";
import { MapPin, Phone, Mail } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-white text-gray-700 pt-14 pb-6">

      <div className="max-w-7xl mx-auto px-10 grid md:grid-cols-4 gap-10">

        {/* Logo + Description */}
        <div>
          <img src="Kidnefy Logo.jpeg" alt="Kidnefy Logo" className="w-25 h-25 object-contain mb-3" />
          <h2 className="text-2xl font-bold mb-4 text-[#274690]">
            Kidnefy
          </h2>

          {/* <p className="text-gray-500 text-sm leading-relaxed mb-4">
            Pioneering the future of renal care through artificial intelligence.
            We are dedicated to making kidney disease detection early,
            accurate, and accessible for everyone.
          </p> */}
        </div>

        {/* Links */}
        <div>
          <h3 className="font-semibold mb-4 text-[#274690]">Links</h3>
          <ul className="space-y-2 text-gray-500 text-sm">
            <li className="hover:text-[#274690] cursor-pointer">Home</li>
            <li className="hover:text-[#274690] cursor-pointer">Kidney Check</li>
            <li className="hover:text-[#274690] cursor-pointer">Risk Detector</li>
            <li className="hover:text-[#274690] cursor-pointer">Refrences</li>
            <li className="hover:text-[#274690] cursor-pointer">About Us</li>
          </ul>
        </div>

        {/* Research */}
        <div>
          <h3 className="font-semibold mb-4 text-[#274690]">Research</h3>
          <ul className="space-y-2 text-gray-500 text-sm">
            <li className="hover:text-[#274690] cursor-pointer">Dataset Information</li>
            <li className="hover:text-[#274690] cursor-pointer">Model Performance</li>
            <li className="hover:text-[#274690] cursor-pointer">Research Papers</li>
            <li className="hover:text-[#274690] cursor-pointer">Project Documentation</li>
          </ul>
        </div>

        {/* Contact */}
        <div>
          <h3 className="font-semibold mb-4 text-[#274690]">Contact</h3>

          <div className="space-y-3 text-gray-500 text-sm">

            <div className="flex items-start gap-2">
              <MapPin size={16} className="text-[#274690]" />
              <p>Culture and Science City - 6th October</p>
            </div>

            <div className="flex items-center gap-2">
              <Mail size={16} className="text-[#274690]" />
              <p>@Kidnefy.com</p>
            </div>

            {/* <div className="flex items-center gap-2">
              <Phone size={16} className="text-[#274690]" />
              <p>+1 (800) 123-4567</p>
            </div> */}

            <div className="flex items-start gap-2 text-[#274690]">
            <FaLinkedin size={20} className="cursor-pointer" />
            <FaGithub size={20} className="cursor-pointer" />
            </div>
          </div>
         </div>
        </div>

      {/* Bottom */}
      
      <div className="border-t border-gray-200 mt-2 pt-4 text-center text-sm text-gray-500">
        &copy; 2026 Kidnefy. All rights reserved.
      </div>

    </footer>
  );
}
