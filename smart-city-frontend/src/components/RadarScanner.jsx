import React from "react";
import "./radar.css";

export default function RadarScanner() {
  return (
    <div className="radar-container">
      <div className="radar">
        <div className="line"></div>
        <div className="ping ping1"></div>
        <div className="ping ping2"></div>
      </div>
    </div>
  );
}
