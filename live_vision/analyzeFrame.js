const Anthropic = require("@anthropic-ai/sdk");

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const SYSTEM_PROMPT = `You are a vision analysis component in a 911 emergency response system. You will be shown ONE frame from a live bystander video feed. Your job is to extract ONLY what is visibly verifiable in this single frame.

CRITICAL RULES:
Do not infer what happened or why. Only report what you can see right now.
If something is ambiguous, occluded, or you're not sure, say so, do not guess.
If the frame is blurry, dark, poorly framed, or doesn't show a clear scene, set frame_quality accordingly and lower your confidence score. Do not try to compensate by inferring more than the frame supports.
This output will be combined with other frames over time by a separate reasoning step. Your only job is accurate single frame observation.

Respond with ONLY valid JSON matching the schema, no markdown fences, no preamble.

Required schema:
{
  "people_count": <integer>,
  "injury_visible": <boolean>,
  "injury_severity_estimate": <"none"|"low"|"moderate"|"high"|"unknown">,
  "injury_location": <string or null>,
  "bleeding_visible": <boolean>,
  "bleeding_severity_estimate": <"none"|"low"|"moderate"|"high"|"unknown">,
  "smoke_visible": <boolean>,
  "fire_visible": <boolean>,
  "person_motion": <"moving"|"still"|"unknown">,
  "person_responsive": <"responsive"|"unresponsive"|"unknown">,
  "hazards": <array from: fire,smoke,broken_glass,structural_damage,downed_power_line,vehicle,weapon_visible,water_hazard,chemical_spill>,
  "frame_quality": <"usable"|"blurry"|"dark"|"obstructed"|"no_scene">,
  "quality_issues": <array of strings, possibly empty>,
  "confidence": <0.0-1.0>,
  "notes": <string, max ~20 words, factual, no speculation>
}`;

async function analyzeFrame(base64ImageData, mimeType = "image/jpeg") {
  const response = await client.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 1024,
    system: SYSTEM_PROMPT,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: mimeType,
              data: base64ImageData,
            },
          },
          {
            type: "text",
            text: "Analyze this frame.",
          },
        ],
      },
    ],
  });

  const raw = response.content[0].text.trim();
  return JSON.parse(raw);
}

module.exports = { analyzeFrame };
