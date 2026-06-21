require("dotenv").config({ path: require("path").resolve(__dirname, "../.env") });
const express = require("express");
const path = require("path");
const os = require("os");
const localtunnel = require("localtunnel");
const Anthropic = require("@anthropic-ai/sdk");

function getLocalIP() {
  for (const ifaces of Object.values(os.networkInterfaces())) {
    for (const iface of ifaces) {
      if (iface.family === "IPv4" && !iface.internal) return iface.address;
    }
  }
  return "localhost";
}
const { analyzeFrame } = require("./analyzeFrame");

const app = express();
const PORT = 3000;
const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

app.use(express.json({ limit: "10mb" }));
app.use(express.static(path.join(__dirname, "public")));

let publicUrl = null;

app.get("/server-url", (req, res) => {
  res.json({ url: publicUrl || `http://${getLocalIP()}:${PORT}` });
});

app.post("/analyze", async (req, res) => {
  const { frame, mimeType, model } = req.body;

  if (!frame) {
    return res.status(400).json({ error: "Missing frame data" });
  }

  const timestamp = new Date().toISOString();

  try {
    const modelId = model === "haiku"
      ? "claude-haiku-4-5-20251001"
      : "claude-sonnet-4-6";
    const result = await analyzeFrame(frame, mimeType || "image/jpeg", modelId);
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
    const raw = response.content[0].text.trim();
    const fenceMatch = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
    const result = JSON.parse(fenceMatch ? fenceMatch[1].trim() : raw);
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
    const raw = response.content[0].text.trim();
    const fenceMatch = raw.match(/```(?:json)?\s*([\s\S]*?)```/);
    const result = JSON.parse(fenceMatch ? fenceMatch[1].trim() : raw);
    console.log(`[translate] "${text.slice(0, 40)}…" → "${result.translation?.slice(0, 40)}…"`);
    res.json(result);
  } catch (err) {
    console.error("[translate] error:", err.message);
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, async () => {
  console.log(`Server running at http://localhost:${PORT}`);

  if (process.env.PUBLIC_URL) {
    publicUrl = process.env.PUBLIC_URL;
    console.log(`Public URL (ngrok): ${publicUrl}`);
    return;
  }

  try {
    const tunnel = await localtunnel({ port: PORT });
    publicUrl = tunnel.url;
    console.log(`Public URL (tunnel): ${tunnel.url}`);
    tunnel.on("close", () => console.log("Tunnel closed — restart server to get a new one"));
    tunnel.on("error", (err) => console.warn("Tunnel error:", err.message));
  } catch (err) {
    console.warn("Could not open public tunnel:", err.message);
    console.log(`Falling back to local IP: http://${getLocalIP()}:${PORT}`);
  }
});
