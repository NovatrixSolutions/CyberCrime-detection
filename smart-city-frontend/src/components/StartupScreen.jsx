import { motion } from "framer-motion";
import { useEffect, useState } from "react";

export default function StartupScreen({ onFinish }) {
  const messages = [
    "🔍 Initializing Smart City AI Firewall...",
    "🛰 Activating IoT Traffic Monitor...",
    "🧠 Loading Reinforcement Learning Engine...",
    "🛡 Deploying Cyber Defense Protocols...",
    "✅ System Ready"
  ];

  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (index < messages.length - 1) {
      const timer = setTimeout(() => setIndex(index + 1), 1600);
      return () => clearTimeout(timer);
    } else {
      setTimeout(() => onFinish(), 1200);
    }
  }, [index]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-black text-green-300 text-xl font-mono tracking-wide">
      <motion.div
        key={index}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        {messages[index]}
      </motion.div>
    </div>
  );
}
