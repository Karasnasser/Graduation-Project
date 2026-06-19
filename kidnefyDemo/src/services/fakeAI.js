export async function analyzeImage(file, useRealAPI = false) {

  if (!file) {
    throw new Error("No image file provided");
  }

  try {

    if (useRealAPI) {

      const base64 = await toBase64(file);

      const res = await fetch("https://api.example.com/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          user_id: 1,
          image_base64: base64
        })
      });

      if (!res.ok) {
        throw new Error("API request failed");
      }

      const data = await res.json();

      return {
        risk: Math.round((data.score || 0.5) * 100),
        stage: data.stage || "G2",
        level: data.level || "Moderate",
        recommendations:
          data.notes || "Follow up with a medical professional."
      };

    } else {

      return fakeAI(file);

    }

  } catch (error) {

    console.error("AI analyze error:", error);

    return {
      risk: 0,
      stage: "G1",
      level: "Low",
      recommendations: "Analysis failed. Please try again."
    };

  }
}


// ----------------------------
// Fake AI Simulation
// ----------------------------

function fakeAI(file) {

  return new Promise((resolve) => {

    const sizeKB = Math.max(1, Math.round(file.size / 1024));

    setTimeout(() => {

      const base = Math.random() * 0.6 + 0.2;
      const modifier = Math.min(0.2, (sizeKB / 5000) * 0.2);

      const score = Math.min(0.99, base + modifier);
      const risk = Math.round(score * 100);

      let stage = "G1";

      if (risk > 80) stage = "G5";
      else if (risk > 65) stage = "G4";
      else if (risk > 50) stage = "G3";
      else if (risk > 30) stage = "G2";

      let level = "Low";

      if (risk > 70) level = "High";
      else if (risk > 40) level = "Moderate";

      const recommendations = {
        Low: "Kidney function appears normal. Maintain a healthy lifestyle.",
        Moderate: "Monitor kidney function and consult a doctor.",
        High: "High risk detected. Immediate nephrologist consultation recommended."
      };

      resolve({
        risk,
        stage,
        level,
        recommendations: recommendations[level]
      });

    }, 1500);

  });

}


// ----------------------------
// Convert Image to Base64
// ----------------------------

function toBase64(file) {

  return new Promise((resolve, reject) => {

    const reader = new FileReader();

    reader.readAsDataURL(file);

    reader.onload = () => {
      const base64 = reader.result.split(",")[1];
      resolve(base64);
    };

    reader.onerror = (error) => reject(error);

  });

}