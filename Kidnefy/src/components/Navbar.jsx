import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import axios from "axios";
import { Bell, Menu, X, Home, Activity, Info, BookOpen, Stethoscope } from "lucide-react";
import { motion } from "framer-motion";

export default function Navbar() {
  const loc = useLocation();
  const [user, setUser] = useState(null);
  const [profileOpen, setProfileOpen] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { name: "Home", path: "/", icon: Home },
    { name: "Kidney Check", path: "/Predict", icon: Activity },
    { name: "Risk Detector", path: "/SmartAlerts", icon: Stethoscope },
    { name: "References", path: "/references", icon: BookOpen },
    { name: "About Us", path: "/about", icon: Info },
  ];

  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem("token");
      if (!token) return;
      try {
        const res = await axios.get("http://localhost:8000/api/user", {
          headers: { Authorization: `Bearer ${token}` },
        });
        setUser(res.data);
      } catch {
        setUser(null);
      }
    };
    fetchUser();
  }, []);

  useEffect(() => {
    const fetchNotifications = async () => {
      const token = localStorage.getItem("token");
      if (!token) return;
      try {
        const res = await axios.get(
          "http://localhost:8000/api/notifications",
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        setNotifications(res.data);
      } catch {}
    };
    fetchNotifications();
  }, []);

  const logout = () => {
    localStorage.removeItem("token");
    setUser(null);
  };

  return (
    <nav className="bg-blue-900 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 md:px-8 py-3 flex items-center justify-between">

        {/* Logo */}
        <div className="flex items-center gap-2 ml-1">
          
          <div ></div>

          <Link to="/" className="flex items-center gap-2">
            <img src="Kidnefy_Logo_white.png" alt="Kidnefy Logo" className="w-10 object-contain" />
            <h1
              className="
                font-extrabold
                text-2xl
                text-white
                cursor-pointer
              "
            >
              Kidnefy
            </h1>
          </Link>

        </div>

        {/* Desktop Links */}
        <motion.ul
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="hidden md:flex gap-8 text-white text-sm"
        >
          {navLinks.map((link) => {
            const Icon = link.icon;
            return (
              <motion.li
                key={link.path}
                whileHover={{ y: -2 }}
                transition={{ type: "spring", stiffness: 260, damping: 18 }}
              >
                <Link
                  to={link.path}
                  className={`inline-flex items-center gap-2 ${
                    loc.pathname === link.path
                      ? "text-white font-bold"
                      : "text-gray-200 hover:text-gray-100"
                  } transition-colors duration-200`}
                >
                  <Icon size={16} />
                  {link.name}
                </Link>
              </motion.li>
            );
          })}
        </motion.ul>

        {/* Right Section */}
        <div className="flex items-center gap-2 md:gap-4">

          {/* Notifications */}
          {user && (
            <div className="relative">
              <button
                onClick={() => setNotificationsOpen(!notificationsOpen)}
                className="p-2 rounded-full hover:bg-white/10 transition"
              >
                <Bell
                  size={20}
                  className={notifications.length > 0 ? "text-white animate-pulse" : "text-white"}
                />
                {notifications.length > 0 && (
                  <span className="absolute top-0 right-0 text-xs bg-red-500 text-white w-5 h-5 flex items-center justify-center rounded-full">
                    {notifications.length}
                  </span>
                )}
              </button>

              <div
                className={`absolute right-0 mt-2 w-72 sm:w-80 bg-white shadow-lg rounded-lg overflow-hidden transition ${
                  notificationsOpen ? "block" : "hidden"
                }`}
              >
                <div className="p-3 font-semibold border-b">
                  Notifications
                </div>
                {notifications.length === 0 ? (
                  <p className="p-3 text-sm text-gray-500">
                    No notifications
                  </p>
                ) : (
                  notifications.map((n, i) => (
                    <div key={i} className="p-3 border-b text-sm">
                      {n.message}
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* Guest */}
          {!user && (
              <motion.div
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="hidden sm:flex"
              >
               <Link to="/signin" className="relative overflow-hidden px-6 py-2 rounded-3xl text-sm font-bold text-blue-900 bg-white/90 backdrop-blur-md border border-white/40 shadow-[0_8px_25px_rgba(0,0,0,0.12)]hover:shadow-[0_12px_35px_rgba(59,130,246,0.25)] hover:scale-105 transition-all duration-300">
  
                  <span className="relative z-10 flex items-center gap-2">
                   Sign in
                  </span>

                  {/* Shine Animation */}
                  <motion.span
                    initial={{ x: "-120%" }}
                    whileHover={{ x: "220%" }}
                    transition={{ duration: 0.8 }}
                    className="
                      absolute
                      top-0
                      left-0
                      w-1/3
                      h-full
                      bg-white/20
                      skew-x-12
                    "
                  />
                </Link>
              </motion.div>
            )}

          {/* User */}
          {user && (
            <div className="relative hidden sm:block">
              <div
                onClick={() => setProfileOpen(!profileOpen)}
                className="flex items-center gap-2 bg-[#708fd1] text-white px-3 py-1.5 rounded-lg cursor-pointer"
              >
                <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center text-[#2e4b8b] font-bold">
                  {user.name?.charAt(0)}
                </div>
                <span className="text-sm">{user.name}</span>
              </div>

              <div
                className={`absolute right-0 mt-2 bg-white shadow-lg rounded-lg w-40 ${
                  profileOpen ? "block" : "hidden"
                }`}
              >
                <Link className="block px-4 py-2 hover:bg-gray-100" to="/profile">
                  Profile
                </Link>
                <Link className="block px-4 py-2 hover:bg-gray-100" to="/settings">
                  Settings
                </Link>
                <button
                  onClick={logout}
                  className="w-full text-left px-4 py-2 text-red-500 hover:bg-gray-100"
                >
                  Logout
                </button>
              </div>
            </div>
          )}

          {/* Mobile Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2"
          >
            {mobileMenuOpen ? <X /> : <Menu />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={mobileMenuOpen ? { opacity: 1, y: 0 } : { opacity: 0, y: -10 }}
        transition={{ duration: 0.24 }}
        className={`md:hidden bg-white border-t overflow-hidden ${
          mobileMenuOpen ? "block" : "hidden"
        }`}
      >
        <div className="flex flex-col p-4 gap-3">
          {navLinks.map((link) => {
            const Icon = link.icon;
            return (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => setMobileMenuOpen(false)}
                className="py-2 border-b flex items-center gap-2 text-gray-700 hover:text-blue-900 transition-colors"
              >
                <Icon size={16} />
                {link.name}
              </Link>
            );
          })}

          {!user && (
            <>
              <Link to="/signin" className="py-2">
                login
              </Link>
              
            </>
          )}

          {user && (
            <>
              <Link to="/profile" className="py-2">
                Profile
              </Link>
              <button onClick={logout} className="text-left py-2 text-red-500">
                Logout
              </button>
            </>
          )}
        </div>
      </motion.div>
    </nav>
  );
}