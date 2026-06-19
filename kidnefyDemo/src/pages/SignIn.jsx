import React, { useState } from "react";
import loginImg from "../assets/login-medical1.jpg";
import api from "../api/api";
import { Link } from "react-router-dom";
import { Mail, Lock } from "lucide-react";

export default function SignIn() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const res = await api.post("/login", { email, password });

      localStorage.setItem("token", res.data.token);
      window.location.href = "/";
    } catch (err) {
      setError(
        err.response?.data?.message ||
          "Email or password is incorrect"
      );
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-linear-to-b from-blue-900 to-white p-6">
      <div className="flex flex-col items-center">
        <div className="flex bg-white rounded-3xl shadow-2xl overflow-hidden max-w-4xl w-full">
          {/* Left Image */}
          <div className="hidden md:block md:w-1/2">
            <img
              src={loginImg}
              alt="Medical AI"
              className="h-full w-full object-cover"
            />
          </div>

          {/* Right Form */}
          <div className="w-full md:w-[50%] p-8 flex flex-col justify-center">
            <div className="text-center mb-6">
              <h1 className="text-2xl font-bold text-blue-900">
                Welcome Back
              </h1>
              <p className="text-sm text-gray-500 mt-1">Sign in to your account</p>
            </div>

            {error && (
              <p className="text-red-500 text-sm text-center mb-3">
                {error}
              </p>
            )}

            <form className="space-y-5" onSubmit={handleSubmit}>
              {/* Email */}
              <div className="relative">
                <Mail
                  className="absolute left-3 top-3.5 text-blue-600"
                  size={18}
                />

                <input
                  type="email"
                  placeholder="Enter your Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full border-2 border-gray-200 rounded-xl pl-10 pr-4 py-3.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition placeholder:text-gray-400"
                  required
                />
              </div>

              {/* Password */}
              <div className="relative">
                <Lock
                  className="absolute left-3 top-3.5 text-blue-600"
                  size={18}
                />

                <input
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full border-2 border-gray-200 rounded-xl pl-10 pr-10 py-3.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition placeholder:text-gray-400"
                  required
                />

                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-4 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M13.875 18.825A10.05 10.05 0 0112 19c-5 0-9-5-9-5a14.134 14.134 0 014.174-4.83m1.451-1.45A9.953 9.953 0 0112 5c5 0 9 5 9 5a14.134 14.134 0 01-2.175 2.73m-3.4 2.63l-4-4m0 0l-4-4m4 4l4 4"
                      />
                    </svg>
                  ) : (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-5 w-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.477 0 8.268 2.943 9.542 7-1.274 4.057-5.065 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                      />
                    </svg>
                  )}
                </button>
              </div>

              {/* Remember & Forgot */}
              <div className="flex justify-between items-center text-sm">
                <label className="flex items-center gap-2 text-gray-600 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={() => setRememberMe(!rememberMe)}
                    className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  Remember me
                </label>

                <a
                  href="#"
                  className="text-blue-800 hover:underline font-medium"
                >
                  Forgot Password?
                </a>
              </div>

              {/* Login Button */}
              <button
                type="submit"
                className="w-full bg-linear-to-r from-blue-900 to-blue-800 text-white py-3.5 rounded-xl hover:from-blue-800 hover:to-blue-700 transition shadow-lg font-semibold text-base mt-2 tracking-wide"
              >
                Sign in
              </button>
            </form>

            <p className="text-center text-sm text-gray-500 mt-5">
              Don't have an account?
              <Link
                to="/signup"
                className="text-blue-800 ml-1 hover:underline font-medium"
              >
                Sign Up
              </Link>
            </p>
          </div>
        </div>

        {/* Back to Home Link */}
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-6 py-3 mt-8 bg-white rounded-full shadow-md hover:shadow-lg border border-blue-100 text-blue-800 font-medium text-sm transition-all duration-300 hover:-translate-y-0.5 group"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4 transition-transform duration-300 group-hover:-translate-x-1"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M10 19l-7-7m0 0l7-7m-7 7h18"
            />
          </svg>
          Back to Home
        </Link>
      </div>
    </div>
  );
}