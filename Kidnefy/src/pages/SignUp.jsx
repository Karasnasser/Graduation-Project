import React, { useState } from "react";
import api from "../api/api";
import { Link } from "react-router-dom";
import { User, Mail, Lock, Phone } from "lucide-react";
import registerImg from "../assets/login-medical.jpg";

export default function SignUp() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    gender: "",
    phone: "",
    password: "",
    confirmPassword: "",
  });

  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    try {
      const res = await api.post("/register", {
        name: formData.name,
        email: formData.email,
        gender: formData.gender,
        phone: formData.phone,
        password: formData.password,
      });

      // تسجيل ناجح
      window.location.href = "/signin";
    } catch (err) {
      console.log("Register Error Full Response:", err.response?.data);

      // أولاً: لو message موجودة
      if (err.response?.data?.message) {
        setError(err.response.data.message);
      } 
      // ثانياً: لو فيه errors array من Laravel
      else if (err.response?.data?.errors) {
        const firstError = Object.values(err.response.data.errors)[0][0];
        setError(firstError);
      } 
      // أي خطأ غير معروف
      else {
        setError("Registration failed");
      }
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 p-6">
      <div className="flex bg-white rounded-3xl shadow-xl overflow-hidden max-w-5xl w-full">
        {/* Left Image */}
        <div className="hidden md:block md:w-1/2">
          <img
            src={registerImg}
            alt="Medical AI"
            className="h-full w-full object-cover"
          />
        </div>

        {/* Right Form */}
        <div className="w-full md:w-1/2 p-10">
          <div className="text-center mb-6">
            <h1 className="text-2xl font-bold text-blue-900">Create Account</h1>
            {/* <p className="text-gray-500 text-sm mt-1">
              Create your account to manage diabetic nephropathy cases with AI
              assistance.
            </p> */}
          </div>

          {error && (
            <p className="text-red-500 text-sm mb-3 text-center">{error}</p>
          )}

          <form className="space-y-5" onSubmit={handleSubmit}>
            {/* First Name & Second Name Row */}
            <div className="grid grid-cols-2 gap-4">
              <div className="relative">
                <User className="absolute left-3 top-3.5 text-blue-600" size={18} />
                <input
                  type="text"
                  name="firstName"
                  placeholder="First Name"
                  className="w-full border-2 border-gray-200 rounded-xl pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  required
                />
              </div>
              <div className="relative">
                <User className="absolute left-3 top-3.5 text-blue-600" size={18} />
                <input
                  type="text"
                  name="secondName"
                  placeholder="Second Name"
                  className="w-full border-2 border-gray-200 rounded-xl pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  required
                />
              </div>
            </div>

            {/* Email */}
            <div className="relative">
              <Mail className="absolute left-3 top-3.5 text-blue-600" size={18} />
              <input
                type="email"
                name="email"
                placeholder="name@example.com"
                className="w-full border-2 border-gray-200 rounded-xl pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                required
              />
            </div>

            {/* Phone */}
            <div className="relative">
              <Phone className="absolute left-3 top-3.5 text-blue-600" size={18} />
              <input
                type="tel"
                name="phone"
                placeholder="Phone Number"
                className="w-full border-2 border-gray-200 rounded-xl pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                required
              />
            </div>

            {/* Gender */}
            {/* <div className="relative">
              <User className="absolute left-3 top-3.5 text-blue-600" size={18} />
              <select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                className="w-full border-2 border-gray-200 rounded-xl pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition appearance-none bg-white cursor-pointer"
                required
              >
                <option value="" disabled hidden className="text-gray-400">
                  Select Gender
                </option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div> */}

            {/* Password */}
            <div className="relative">
              <Lock className="absolute left-3 top-3.5 text-blue-600" size={18} />
              <input
                type="password"
                name="password"
                placeholder="Password (at least 8 characters)"
                className="w-full border-2 border-gray-200 rounded-xl pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                required
              />
            </div>

            {/* Confirm Password */}
            <div className="relative">
              <Lock className="absolute left-3 top-3.5 text-blue-600" size={18} />
              <input
                type="password"
                name="confirmPassword"
                placeholder="Confirm Password"
                className="w-full border-2 border-gray-200 rounded-xl pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                required
              />
            </div>

            {/* Button */}
            <button
              type="submit"
              className="w-full bg-linear-to-r from-blue-900 to-blue-800 text-white py-3 rounded-xl hover:from-blue-800 hover:to-blue-700 transition shadow-lg font-semibold text-lg mt-6"
            >
              Sign Up
            </button>
          </form>

          {/* Login Link */}
          <p className="text-center text-sm text-gray-500 mt-5">
            Already have an account?
            <Link to="/signin" className="text-blue-800 ml-1 hover:underline">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}