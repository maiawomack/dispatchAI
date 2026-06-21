require("dotenv").config();
const express = require("express");
const path = require("path");
const { analyzeFrame } = require("./analyzeFrame");

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "10mb" }));
app.use(express.static(path.join(__dirname, "public")));

app.post("/analyze", async (req, res) => {
  const { frame, mimeType } = req.body;

  if (!frame) {
    return res.status(400).json({ error: "Missing frame data" });
  }

  const timestamp = new Date().toISOString();

  try {
    const result = await analyzeFrame(frame, mimeType || "image/jpeg");
    result.timestamp = timestamp;
    console.log(`[${timestamp}] frame_quality=${result.frame_quality} confidence=${result.confidence}`);
    res.json({ ok: true, result });
  } catch (err) {
    console.error(`[${timestamp}] analysis error:`, err.message);
    // Return a degraded result rather than crashing the loop
    res.json({
      ok: false,
      error: err.message,
      result: {
        frame_quality: "no_scene",
        confidence: 0,
        quality_issues: ["analysis_error"],
        notes: "Frame analysis failed",
        timestamp,
      },
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Open http://localhost:${PORT} in a browser to start capture`);
});
