require("dotenv").config();
const express = require("express");
const path = require("path");
const Anthropic = require("@anthropic-ai/sdk");
const { analyzeFrame } = require("./analyzeFrame");

const app = express();
const PORT = 3000;
const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

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

app.post("/detect-language", async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: "Missing text" });

  try {
    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 128,
      messages: [{
        role: "user",
        content: `Identify the language of the following text. Respond with ONLY valid JSON, no markdown: {"language": "<full language name in English>", "language_code": "<ISO 639-1 code>", "confidence": <0.0-1.0>}\n\nText: ${JSON.stringify(text)}`,
      }],
    });
    const result = JSON.parse(response.content[0].text.trim());
    console.log(`[detect-language] "${text.slice(0, 40)}…" → ${result.language}`);
    res.json(result);
  } catch (err) {
    console.error("[detect-language] error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

app.post("/translate", async (req, res) => {
  const { text, from_language } = req.body;
  if (!text) return res.status(400).json({ error: "Missing text" });

  try {
    const response = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 512,
      messages: [{
        role: "user",
        content: `Translate the following ${from_language || "text"} to English. Respond with ONLY valid JSON, no markdown: {"translation": "<translated text>"}\n\nText: ${JSON.stringify(text)}`,
      }],
    });
    const result = JSON.parse(response.content[0].text.trim());
    console.log(`[translate] "${text.slice(0, 40)}…" → "${result.translation?.slice(0, 40)}…"`);
    res.json(result);
  } catch (err) {
    console.error("[translate] error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Open http://localhost:${PORT} in a browser to start capture`);
});
