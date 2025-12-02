import { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { FaCloudUploadAlt, FaShieldAlt, FaBrain } from "react-icons/fa";
import { Pie, Bar } from "react-chartjs-2";
import Tilt from "react-parallax-tilt";
import RadarScanner from "./components/RadarScanner";
import MatrixRain from "./components/MatrixRain";
import StartupScreen from "./components/StartupScreen";

import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
} from "chart.js";

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement
);

const API_BASE = "http://127.0.0.1:8000";

const speak = (text) => {
  const voice = new SpeechSynthesisUtterance(text);
  voice.pitch = 1;
  voice.rate = 1;
  voice.volume = 1;
  window.speechSynthesis.speak(voice);
};

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trainInfo, setTrainInfo] = useState(null);
  const [predictInfo, setPredictInfo] = useState(null);
  const [error, setError] = useState("");
  const [bootComplete, setBootComplete] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError("");
  };

  const runFullPipeline = async () => {
    if (!file) {
      setError("Please choose a CSV dataset first.");
      return;
    }

    setLoading(true);
    setError("");
    setTrainInfo(null);
    setPredictInfo(null);

    try {
      // 1) upload-dataset
      const formData = new FormData();
      formData.append("file", file);

      await axios.post(`${API_BASE}/upload-dataset`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      // 2) preprocess
      await axios.get(`${API_BASE}/preprocess`);

      // 3) prepare-data
      const prepRes = await axios.get(`${API_BASE}/prepare-data`);

      // 4) init-dddqn
      await axios.get(`${API_BASE}/init-dddqn`);

      // 5) train-dddqn
      const trainRes = await axios.get(`${API_BASE}/train-dddqn`, {
        params: { epochs: 3, batch_size: 64 },
      });

      setTrainInfo({
        accuracy: trainRes.data.test_accuracy_approx,
        feature_count: prepRes.data.feature_count,
        total_rows: prepRes.data.total_rows,
      });

      // 6) predict-dddqn-file (using same file for demo)
      const predForm = new FormData();
      predForm.append("file", file);

      const predRes = await axios.post(
        `${API_BASE}/predict-dddqn-file`,
        predForm,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      setPredictInfo(predRes.data);

      // 🔊 Voice Feedback
      if (predRes.data.threat_count_label_1 > predRes.data.safe_count_label_0) {
        speak("Warning. Network attack detected. Take immediate action.");
      } else if (
        predRes.data.threat_count_label_1 >
        predRes.data.total_rows * 0.2
      ) {
        speak("Caution. Suspicious network activity detected.");
      } else {
        speak("System secure. No threats detected.");
      }
    } catch (err) {
      console.error(err);
      setError(
        err.response?.data?.detail ||
          "Something went wrong while contacting the backend."
      );
    } finally {
      setLoading(false);
    }
  };

  // Chart data
  const pieData =
    predictInfo &&
    (() => {
      const safe = predictInfo.safe_count_label_0 || 0;
      const threat = predictInfo.threat_count_label_1 || 0;
      return {
        labels: ["Safe (0)", "Threat (1)"],
        datasets: [
          {
            data: [safe, threat],
            backgroundColor: ["#22c55e", "#ef4444"],
            hoverOffset: 6,
          },
        ],
      };
    })();

  const barData =
    trainInfo &&
    (() => {
      return {
        labels: ["DDDQN Accuracy"],
        datasets: [
          {
            label: "Accuracy",
            data: [Math.round((trainInfo.accuracy || 0) * 100)],
            backgroundColor: "#6366f1",
          },
        ],
      };
    })();

  // 🔥 Threat ratio + level for card UI
  const threatRatio =
    predictInfo && predictInfo.total_rows
      ? predictInfo.threat_count_label_1 / predictInfo.total_rows
      : 0;

  const threatLevel = !predictInfo
    ? "idle"
    : threatRatio > 0.5
    ? "high"
    : threatRatio > 0.2
    ? "medium"
    : "low";

  if (!bootComplete) {
    return <StartupScreen onFinish={() => setBootComplete(true)} />;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white flex flex-col items-center px-4 py-6">
      <MatrixRain />

      {/* Glow background */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute -top-32 -left-32 w-72 h-72 bg-purple-600/40 blur-3xl rounded-full" />
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-cyan-500/30 blur-3xl rounded-full" />
      </div>

      {/* Centered header */}
      <section className="w-full h-screen flex justify-center items-center">
        <header className="w-full flex justify-center mt-6 mb-10">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="neon-title text-3xl md:text-5xl font-extrabold tracking-tight">
              Smart City <span className="text-cyan-400">Cybersecurity</span>{" "}
              Dashboard
            </h1>

            <p className="text-slate-300 mt-3 text-sm md:text-base">
              AI-powered Dueling Double Deep Q-Network with Prioritized
              Experience Replay for real-time Safe vs Threat detection in smart
              city IoT traffic.
            </p>

            {/* Engine logo + text beside each other */}
            <div className="flex items-center justify-center gap-3 mt-4">
              <div className="p-3 rounded-2xl bg-slate-900/70 border border-slate-700 shadow-lg shadow-cyan-500/40">
                <FaBrain className="text-2xl text-cyan-400" />
              </div>
              <div className="text-left text-xs md:text-sm text-slate-200">
                <div className="font-semibold text-cyan-400">
                  DDDQN + PER Threat Engine
                </div>
                <div>Reinforcement Learning Intrusion Detector</div>
              </div>
            </div>
          </div>
        </header>
      </section>

      <main className="w-full max-w-6xl flex flex-col gap-6">
        {/* Upload + Status Section (clean grid) */}
        <motion.div
          className="grid md:grid-cols-3 gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          {/* Upload Card */}
          <div className="md:col-span-2 rounded-3xl bg-slate-900/80 border border-slate-700/80 shadow-2xl shadow-purple-500/20 p-5 md:p-6 space-y-4">
            <div className="flex items-center gap-2">
              <FaCloudUploadAlt className="text-xl text-cyan-400" />
              <h2 className="text-lg md:text-xl font-semibold">
                Upload IoT / Network Traffic Dataset
              </h2>
            </div>

            <p className="text-slate-300 text-xs md:text-sm">
              Upload a CSV file from your smart city / IoT intrusion dataset.
              The system will automatically preprocess, train the DDDQN model,
              and classify traffic as Safe or Threat.
            </p>

            <label className="relative flex items-center justify-center h-28 rounded-2xl border-2 border-dashed border-slate-600 hover:border-cyan-400 cursor-pointer transition-all bg-slate-950/70">
              <input
                type="file"
                accept=".csv"
                className="hidden"
                onChange={handleFileChange}
              />
              <div className="flex flex-col items-center gap-1 text-slate-300">
                <span className="text-sm font-medium">
                  {file ? file.name : "Click to choose CSV dataset"}
                </span>
                <span className="text-[11px] text-slate-400">
                  Your file stays local and goes only to your FastAPI backend.
                </span>
              </div>
            </label>

            {error && (
              <div className="text-xs text-red-400 bg-red-950/40 border border-red-700/60 rounded-xl px-3 py-2">
                {error}
              </div>
            )}

            <button
              onClick={runFullPipeline}
              disabled={loading}
              className="cyber-btn inline-flex items-center gap-2 px-5 py-2.5 rounded-2xl bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-400 hover:to-purple-400 disabled:opacity-50 text-sm font-semibold shadow-lg shadow-cyan-500/30 transition-transform"
            >
              {loading ? "Processing..." : "Run Full Analysis (Train + Predict)"}
            </button>

            <p className="text-[11px] text-slate-400">
              Pipeline: upload → preprocess → prepare features → init DDDQN →
              train → predict (file).
            </p>
          </div>

          {/* Status Card - Glow, Threat Bar, Circular Gauge */}
          <motion.div
            className={
              "rounded-3xl bg-slate-900/80 border p-5 flex flex-col gap-4 shadow-xl transition-transform glow-card " +
              (threatLevel === "high"
                ? "border-red-500 shadow-red-500/40"
                : threatLevel === "medium"
                ? "border-yellow-400 shadow-yellow-400/40"
                : threatLevel === "low"
                ? "border-emerald-400 shadow-emerald-400/40"
                : "border-slate-700/80 shadow-cyan-500/20")
            }
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.99 }}
          >
            {/* Title */}
            <div className="flex items-center gap-2">
              <FaShieldAlt className="text-lg text-emerald-400" />
              <span className="font-semibold text-sm">
                Model Status & Metrics
              </span>
            </div>

            {/* Metrics box */}
            <div className="space-y-2 text-xs text-slate-300 bg-slate-800/60 border border-slate-700 rounded-xl p-3">
              <div className="flex justify-between">
                <span className="text-slate-400">Backend:</span>
                <span className="text-emerald-400 font-medium">Online</span>
              </div>

              <div className="flex justify-between">
                <span className="text-slate-400">DDDQN Accuracy:</span>
                {trainInfo?.accuracy !== undefined ? (
                  <span className="text-cyan-400 font-medium">
                    {(trainInfo.accuracy * 100).toFixed(2)}%
                  </span>
                ) : (
                  <span className="text-slate-500">Not trained</span>
                )}
              </div>

              <div className="flex justify-between">
                <span className="text-slate-400">Features Used:</span>
                <span>{trainInfo?.feature_count ?? "N/A"}</span>
              </div>

              <div className="flex justify-between">
                <span className="text-slate-400">Samples Processed:</span>
                <span>{trainInfo?.total_rows ?? "N/A"}</span>
              </div>
            </div>

            {/* Threat level bar */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] text-slate-400">
                <span>Threat Level</span>
                <span
                  className={
                    threatLevel === "high"
                      ? "text-red-400"
                      : threatLevel === "medium"
                      ? "text-yellow-300"
                      : threatLevel === "low"
                      ? "text-emerald-300"
                      : "text-slate-500"
                  }
                >
                  {threatLevel === "high"
                    ? "High"
                    : threatLevel === "medium"
                    ? "Medium"
                    : threatLevel === "low"
                    ? "Low"
                    : "Idle"}
                </span>
              </div>
              <div className="w-full h-2.5 rounded-full bg-slate-800 overflow-hidden border border-slate-700">
                <div
                  className={
                    "h-full transition-all " +
                    (threatLevel === "high"
                      ? "bg-red-500"
                      : threatLevel === "medium"
                      ? "bg-yellow-400"
                      : threatLevel === "low"
                      ? "bg-emerald-400"
                      : "bg-slate-600")
                  }
                  style={{ width: `${Math.min(threatRatio * 100, 100)}%` }}
                />
              </div>
            </div>

            {/* Circular gauge */}
            {predictInfo && (
              <div className="flex flex-col items-center gap-2 text-xs">
                <div className="relative w-24 h-24">
                  {/* Base circle */}
                  <div className="absolute inset-0 rounded-full bg-slate-900 border border-slate-700" />
                  {/* Conic gauge */}
                  <div
                    className="absolute inset-0 rounded-full"
                    style={{
                      background: `conic-gradient(${
                        threatLevel === "high"
                          ? "#ef4444"
                          : threatLevel === "medium"
                          ? "#facc15"
                          : "#22c55e"
                      } ${threatRatio * 360}deg, rgba(15,23,42,0.95) ${
                        threatRatio * 360
                      }deg)`,
                    }}
                  />
                  {/* Inner circle */}
                  <div className="absolute inset-3 rounded-full bg-slate-950 flex flex-col items-center justify-center">
                    <span className="text-sm font-semibold">
                      {Math.round((threatRatio || 0) * 100)}%
                    </span>
                    <span className="text-[9px] text-slate-400">Threat</span>
                  </div>
                </div>

                <div className="text-[11px] text-slate-300 text-center space-y-0.5">
                  <div>
                    Safe:{" "}
                    <span className="text-emerald-400 font-semibold">
                      {predictInfo.safe_count_label_0}
                    </span>{" "}
                    · Threat:{" "}
                    <span className="text-rose-400 font-semibold">
                      {predictInfo.threat_count_label_1}
                    </span>
                  </div>
                  <div>
                    Total Samples:{" "}
                    <span className="text-slate-200 font-semibold">
                      {predictInfo.total_rows}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </motion.div>

        {/* Charts Row */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Pie chart */}
          <motion.div
            className="rounded-3xl bg-slate-900/70 border border-slate-700/80 shadow-xl shadow-cyan-500/20 p-4"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <h3 className="text-sm font-semibold mb-3">
              Safe vs Threat Distribution
            </h3>
            {pieData && predictInfo ? (
              <div className="flex justify-center">
                <div className="w-64 h-64 sm:w-80 sm:h-80 md:w-96 md:h-96 lg:w-[450px] lg:h-[450px]">
                  <Pie
                    data={pieData}
                    options={{
                      // responsive: true,
                      maintainAspectRatio: false,
                    }}
                  />
                </div>
              </div>
            ) : (
              <p className="text-center text-lg font-semibold mt-4 mb-3">
                Run an analysis to view Safe vs Threat distribution.
              </p>
            )}
          </motion.div>

          {/* Bar chart */}
          <motion.div
            className="rounded-3xl bg-slate-900/70 border border-slate-700/80 shadow-xl shadow-purple-500/20 p-4"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.25 }}
          >
            <h3 className="text-sm font-semibold mb-3">
              Model Evaluation (Accuracy)
            </h3>
            {barData ? (
              <Bar data={barData} options={{ scales: { y: { max: 100 } } }} />
            ) : (
              <p className="text-xs text-slate-400">
                Train the model to visualize accuracy.
              </p>
            )}
          </motion.div>
        </div>

        {/* Alert + Radar + Download */}
        {predictInfo && (
          <motion.div
            className={`rounded-3xl p-5 text-center font-bold text-lg border-2 ${
              predictInfo.threat_count_label_1 >
              predictInfo.safe_count_label_0
                ? "border-red-500 text-red-400 bg-red-900/20 shadow-red-500/40 alert-pulse"
                : predictInfo.threat_count_label_1 >
                  predictInfo.total_rows * 0.2
                ? "border-yellow-400 text-yellow-300 bg-yellow-900/20 shadow-yellow-500/40"
                : "border-emerald-400 text-emerald-300 bg-emerald-900/20 shadow-emerald-500/40"
            }`}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            {predictInfo.threat_count_label_1 >
            predictInfo.safe_count_label_0
              ? "⚠️ ALERT: Network Attack Detected! Take Immediate Action."
              : predictInfo.threat_count_label_1 >
                predictInfo.total_rows * 0.2
              ? "⚠ Warning: Suspicious Activity Detected."
              : "🟢 Network Status: Secure — No Major Threats Detected."}
          </motion.div>
        )}

        {predictInfo && <RadarScanner />}

        {predictInfo && (
          <button
            onClick={() => {
              const blob = new Blob(
                [JSON.stringify(predictInfo, null, 2)],
                {
                  type: "application/json",
                }
              );
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = "cybersecurity_analysis_report.json";
              a.click();
            }}
            className="cyber-btn mt-4 px-6 py-2 bg-slate-800 text-cyan-300 border border-cyan-500 rounded-xl self-center"
          >
            📄 Download Analysis Report
          </button>
        )}

        {/* Preview table with Tilt */}
        <Tilt
          glareEnable={true}
          glareMaxOpacity={0.25}
          tiltMaxAngleX={10}
          tiltMaxAngleY={10}
          scale={1.02}
        >
          <motion.div
            className="rounded-3xl bg-slate-900/70 border border-slate-700/80 shadow-xl shadow-slate-800/70 p-4 overflow-x-auto text-xs"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <h3 className="text-sm font-semibold mb-3">
              Sample Prediction Preview
            </h3>
            {predictInfo && predictInfo.preview_first_5 ? (
              <table className="min-w-full border-collapse">
                <thead>
                  <tr>
                    {Object.keys(predictInfo.preview_first_5).map((col) => (
                      <th
                        key={col}
                        className="border-b border-slate-700 px-2 py-1 text-[11px] text-slate-300 text-left"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(
                    predictInfo.preview_first_5[
                      Object.keys(predictInfo.preview_first_5)[0]
                    ]
                  ).map((rowKey) => (
                    <tr key={rowKey} className="hover:bg-slate-800/60">
                      {Object.keys(predictInfo.preview_first_5).map((col) => (
                        <td
                          key={col}
                          className="px-2 py-1 border-b border-slate-800 text-[11px] text-slate-300"
                        >
                          {String(
                            predictInfo.preview_first_5[col][rowKey]
                          ).slice(0, 18)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="text-xs text-slate-400">
                Once predictions are generated, the first 5 rows will appear
                here with the predicted label.
              </p>
            )}
          </motion.div>
        </Tilt>
      </main>

      <footer className="mt-6 text-[11px] text-slate-500">
        Smart City Cybersecurity • DDDQN + PER • FastAPI + React
      </footer>
    </div>
  );
}

export default App;
