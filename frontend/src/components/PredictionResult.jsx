import React from "react";

const PredictionResult = ({ result }) => {
  if (!result) return null;

  return (
    <div className="result-card">
      <h2>Prediction Result</h2>

      <h3
        className={
          result.prediction === "FAKE"
            ? "fake-text"
            : "real-text"
        }
      >
        {result.prediction}
      </h3>

      <p>
        Confidence: {result.confidence.toFixed(2)}%
      </p>
    </div>
  );
};

export default PredictionResult;