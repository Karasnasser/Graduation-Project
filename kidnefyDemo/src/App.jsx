import React from "react";
import {
  BrowserRouter,
  Routes,
  Route,
  useLocation,
} from "react-router-dom";

import Chatpot from "./pages/ChatBot";
import Navbar from "./components/Navbar";
import HeroSection from "./components/HeroSection";
import Predict from "./pages/Predict";
import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";
import AboutPage from "./pages/AboutPage";
import SmartAlerts from "./pages/SmartAlerts";
import References from "./pages/References";
import Footer from "./components/Footer";
import TestimonialsSection from "./components/TestimonialsSection";
import FeaturesSection from "./components/FeaturesSection";
import RiskAssessmentSection from "./components/RiskAssessmentSection";

function AppContent() {
  const location = useLocation();

  const authPages =
    location.pathname === "/signin" ||
    location.pathname === "/signup";

  return (
    <div className="font-sans bg-white text-gray-900 min-h-screen">

      <Navbar />

      <Routes>
        <Route
          path="/"
          element={
            <>
              <HeroSection />
              <FeaturesSection />
              <RiskAssessmentSection/>

              <TestimonialsSection />
              <Footer />
            </>
          }
        />

        <Route path="/kidney-check" element={<Predict />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/risk-detector" element={<SmartAlerts />} />
        <Route path="/references" element={<References />} />

        <Route path="/signin" element={<SignIn />} />
        <Route path="/signup" element={<SignUp />} />
      </Routes>

      {!authPages && <Chatpot />}
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}