import React, { useState } from "react";
import { predictAudio } from "../services/api";
import Loader from "./Loader";
import PredictionResult from "./PredictionResult";

const UploadAudio = () => {
  const [audioFile, setAudioFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setAudioFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!audioFile) {
      alert("Please select an audio file.");
      return;
    }

    try {
      setLoading(true);

      const response = await predictAudio(audioFile);

      setResult(response);
    } catch (error) {
      console.error(error);

      alert("Prediction Failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <h1>Deepfake Audio Detection</h1>

      <input
        type="file"
        accept=".wav,.mp3"
        onChange={handleChange}
      />

      <button onClick={handleUpload}>
        Upload & Analyze
      </button>

      {loading && <Loader />}

      <PredictionResult result={result} />
    </div>
  );
};

export default UploadAudio;