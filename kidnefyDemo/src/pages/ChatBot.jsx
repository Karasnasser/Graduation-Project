import React, { useState, useRef, useEffect } from "react";

export default function Chatpot() {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hello 👋 How can I help you?" }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", text: input };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: input })
      });

      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.reply }
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "⚠️ Error connecting to AI" }
      ]);
    }

    setLoading(false);
  };

  return (
    <>
      {/* Floating Button */}
      <div
        onClick={() => setOpen(!open)}
        className="fixed bottom-4 right-4 sm:bottom-6 sm:right-6 z-9999 cursor-pointer"
      >
        <div className="bg-linear-to-r from-blue-900 to-blue-500 w-16 h-16 rounded-full flex items-center justify-center shadow-2xl hover:scale-110 transition">

    {/* Bot LOGO */}
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 28 28"
      width="32"
      height="32"
      fill="none"
      stroke="white"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      >    
      {/* Head */}
      <rect x="3" y="10" width="22" height="14" rx="4" />
      
      {/* Lock arch */}
      <path d="M9 10V8a5 5 0 0110 0v2" />
      
      {/* Antenna */}
      <line x1="14" y1="10" x2="14" y2="4" />
      <circle cx="14" cy="3" r="1.5" fill="white" stroke="none" />
      
      {/* Eyes */}
      <circle cx="10" cy="16" r="1.8" fill="white" stroke="none" />
      <circle cx="18" cy="16" r="1.8" fill="white" stroke="none" />
      
      {/* Pupils */}
      <circle cx="10.6" cy="15.5" r="0.7" fill="#1a1a2e"     stroke="none" />
      <circle cx="18.6" cy="15.5" r="0.7" fill="#1a1a2e"     stroke="none" />
      
      {/* Mouth */}
      <path d="M10 21 Q14 24 18 21" />
      
      {/* Ears */}
      <rect x="1" y="13" width="2.5" height="5" rx="1.25" />
      <rect x="24.5" y="13" width="2.5" height="5" rx="1.25" />
      </svg>
    </div>
  </div>

      {/* Chat Box */}
      {open && (
        <div
          className="
            fixed 
            bottom-20 right-3 sm:right-6
            w-[95%] sm:w-96
            max-w-100
            h-[80vh] sm:h-auto
            bg-white 
            rounded-2xl 
            shadow-2xl 
            z-9999 
            flex 
            flex-col 
            overflow-hidden
          "
        >

          {/* Header */}
          <div className="bg-blue-900 to-blue-500 text-white p-3 flex justify-between items-center">

            <div className="flex items-center ">
              
              <span className="text-sm font-semibold">Kidnefy Bot</span>
            </div>

            <button onClick={() => setOpen(false)}>✖</button>
          </div>

          {/* Messages */}
          <div className="p-3 flex-1 overflow-y-auto flex flex-col gap-2 text-sm sm:text-[15px]">

            {messages.map((msg, i) => (
              <div
                key={i}
                className={`px-4 py-2.5 rounded-2xl max-w-100% shadow-sm ${
                  msg.role === "user"
                    ? "bg-blue-800 text-white self-end rounded-br-none"
                    : "bg-gray-200 text-gray-800 self-start rounded-bl-none"
                }`}
              >
                {msg.text}
              </div>
            ))}

            {/* Typing Animation */}
            {loading && (
              <div className="flex gap-1 self-start">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></span>
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-150"></span>
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-300"></span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-2 sm:p-3 flex items-center gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="How Can Help You..?"
              className=" flex-1  rounded-full border border-gray-300 p-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition bg-white/70 outline-none shadow-sm"
              onKeyDown={(e) => e.key === "Enter" && sendMessage()}
            />

            <button
              onClick={sendMessage}
              className="bg-blue-900 text-white px-5 py-2 rounded-full hover:bg-blue-800 transition shadow-md"
            >
              ➤
            </button>
          </div>
        </div>
      )}
    </>
  );
}