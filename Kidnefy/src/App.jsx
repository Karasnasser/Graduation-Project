import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Chatpot from "./pages/ChatBot"; 
import Navbar from "./components/Navbar";
import HeroSection from "./components/HeroSection";

import Predict from "./pages/Predict";
import SignIn from "./pages/SignIn";   
import SignUp from "./pages/SignUp";  
import AboutPage from "./pages/AboutPage";
import SmartAlerts from "./pages/SmartAlerts";


import Footer from "./components/Footer";
import TestimonialsSection from "./components/TestimonialsSection";
import FeaturesSection from "./components/FeaturesSection";

export default function App() {
  return (

    <BrowserRouter>
      <div className="font-sans bg-white text-gray-900 min-h-screen">

        <Navbar />

        <Routes>

          {/* Home */}
          <Route
            path="/"
            element={
              <>
                <HeroSection />
                <FeaturesSection />
                <TestimonialsSection/>
                <Footer/>
              </>
            }
          />

          {/* Pages */}
          <Route path="/predict" element={<Predict />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/smart-alerts" element={<SmartAlerts />} />

          {/* Auth */}
          <Route path="/signin" element={<SignIn />} />
          <Route path="/signup" element={<SignUp />} /> 

        </Routes>

        
        <Chatpot/>

      </div>

    </BrowserRouter>
  );
}