import axios from "axios";

// Leaving this as an empty string ensures the request goes to the 
// same domain that is currently serving your website.
const API_URL = ""; 

export const predictAudio = async (audioFile) => {
  const formData = new FormData();

  formData.append("file", audioFile);

  const response = await axios.post(
    `${API_URL}/predict`, // This will resolve to /predict automatically
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return response.data;
};