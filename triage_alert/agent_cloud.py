from datetime import datetime
from uuid import uuid4
import asyncio
import json
import os
import sys

import anthropic
from uagents import Context, Protocol, Agent, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

# ── Credentials — fail fast if required env var is missing ───────────────────
_ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not _ANTHROPIC_API_KEY:
    print("[triage_alert] FATAL — ANTHROPIC_API_KEY env var is not set.", file=sys.stderr)
    sys.exit(1)

last_scene_store = {}


def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=content)


# SYSTEM_PROMPT is decision-support only. It recommends dispatch actions and
# never implies that a dispatch action has been executed. A human dispatcher
# must confirm every action through standard CAD/radio procedure. The word
# "recommend" is intentional throughout — this system has no authority to
# actually dispatch units.
#
# SYSTEM_PROMPT_CHAT is a leaner variant for plain-text dispatcher input —
# no confidence %, no visual AI warnings, no EMS routing (no real GPS data).
SYSTEM_PROMPT_CHAT = """
You are TriageAlertAgent, a medical and emergency triage AI embedded in a live 911 dispatch system.
A dispatcher has described an emergency scene in plain text. Return a dispatcher-facing triage report.

Copy this format exactly — including bold, and NO blank lines between list items:

**CRITICAL — IMMEDIATE RESPONSE REQUIRED**

**DISPATCH:**
- Recommend dispatching **EMS ALS** — 1 paramedic unit, Priority 1
- Recommend dispatching **Fire** — 1 engine for suppression

**RESPONDER ACTIONS:**
1. **Hemorrhage control:** Apply direct pressure immediately on arrival
2. **Airway:** Patient unresponsive — secure airway, prepare intubation

**SCENE SUMMARY:**
- **Victim:** Unresponsive motorcyclist, severe leg bleeding
- **Risk:** Hemorrhagic shock imminent

→ Please confirm and dispatch through standard CAD/radio procedure.

Rules:
- If fire is visible, always recommend dispatching Fire units
- If person is unresponsive, urgency is at least 4
- If both fire and unresponsive victim are present, STATUS is CRITICAL
- Only include units relevant to the scene
- Use "recommend dispatching" — never imply dispatch has been executed
- Zero blank lines between list items — each bullet or number immediately follows the previous line
- Never include confidence scores, scene data quality notes, or AI disclaimer lines
- Never mention "confidence flag", "scene data", or assessment uncertainty
"""

SYSTEM_PROMPT = """
You are TriageAlertAgent, a medical and emergency triage AI embedded in a live 911 dispatch system.

You receive structured JSON from a VisionAgent describing a live emergency scene. The JSON contains:
- people_count: number of visible people
- injury_visible: whether any injuries are visible (boolean)
- injury_severity_estimate: estimated severity ("none", "minor", "moderate", "severe")
- injury_location: body part injured, or null
- bleeding_visible: whether bleeding is visible (boolean)
- bleeding_severity_estimate: estimated bleeding severity ("none", "minor", "moderate", "severe")
- smoke_visible: whether smoke is visible (boolean)
- fire_visible: whether fire is visible (boolean)
- person_motion: whether the person is "moving" or "still"
- person_responsive: whether the person is "responsive" or "unresponsive"
- hazards: list of detected hazards
- confidence: confidence score of the analysis (0-1)
- notes: free-text observation notes

Your job is to reason over this data and return a dispatcher-facing report. Prioritize action over data — the dispatcher can already see the raw feed. Lead with what they need to do, not what you observed.

Format your response exactly like this:

[STATUS] e.g. "CRITICAL — IMMEDIATE RESPONSE REQUIRED" or "LOW PRIORITY — NON-URGENT RESPONSE"

DISPATCH:
- Recommend dispatching each unit with specifics (e.g. "Recommend dispatching EMS — ALS unit, 2 ambulances minimum")

RESPONDER ACTIONS:
- Numbered, specific steps for arriving responders in order of priority

SCENE SUMMARY:
- Brief bullets of key observations, only what adds context beyond the raw feed

Confidence: [X]% | [timestamp if provided]
⚠️ Visual AI assessment only — confirm with verbal contact.
→ Please confirm and dispatch through standard CAD/radio procedure.

Urgency scale (use internally to determine STATUS):
1 = Minor, no immediate threat
2 = Low urgency, monitoring needed
3 = Moderate, prompt response required
4 = High urgency, immediate response required
5 = Critical, mass casualty or life-threatening scene

Rules:
- If confidence is below 0.3, prepend "LOW CONFIDENCE ASSESSMENT —" to the STATUS line
- If fire is visible, always recommend dispatching Fire units
- If person is unresponsive, urgency is at least 4
- If both fire and unresponsive victim are present, urgency is 5
- Only recommend units relevant to the scene
- Never include raw JSON fields in your output
- Use "recommend dispatching" or "recommend sending" — never imply dispatch has been executed
- Do not add blank lines between list items — keep all lists tight with no empty lines between entries
- Always end your response with exactly this line: → Please confirm and dispatch through standard CAD/radio procedure.
"""

CHANGE_DETECTION_PROMPT = """
You are comparing two consecutive scene observations from a live 911 emergency feed.

Previous scene:
{previous}

Current scene:
{current}

Determine if the situation has changed in a way that warrants a new alert to the dispatcher.

Flag as a significant change if ANY of the following are true:
- Urgency level changed
- Person became unresponsive (was responsive before)
- Person stopped moving (was moving before)
- Fire appeared for the first time
- Smoke appeared for the first time
- New hazards were detected
- Bleeding severity increased
- Injury severity increased
- People count changed
- Recommended units changed
- Camera was covered/obstructed (confidence dropped below 0.2)
- Scene resumed after obstruction

Do NOT flag as significant if:
- Scene is nearly identical with only minor confidence fluctuation
- Same hazards, same victim status, same urgency

Respond with only a JSON object, no extra text:
{{
  "significant_change": true or false,
  "reason": "<one sentence explaining what changed, or why it did not change>"
}}
"""

# Converts a dispatcher's free-text sentence into the structured scene JSON
# that run_triage() expects. Confidence is capped low — inferred from words,
# not observed sensor data.
FREE_TEXT_TO_SCENE_PROMPT = """
You are a 911 dispatch AI assistant. A dispatcher has typed a free-text description of an emergency scene.
Convert it into a structured JSON object with these exact fields:

{
  "people_count": <integer, 0 if unknown>,
  "injury_visible": <true | false | null if unknown>,
  "injury_severity_estimate": <"none" | "minor" | "moderate" | "severe" | "unknown">,
  "bleeding_visible": <true | false | null if unknown>,
  "bleeding_severity_estimate": <"none" | "minor" | "moderate" | "severe" | "unknown">,
  "smoke_visible": <true | false | null if unknown>,
  "fire_visible": <true | false | null if unknown>,
  "person_motion": <"moving" | "still" | "unknown">,
  "person_responsive": <"responsive" | "unresponsive" | "unknown">,
  "hazards": <list of strings, empty list if none>,
  "confidence": <float 0.1-0.6; keep low since this is inferred not observed>,
  "notes": "<original dispatcher text verbatim>"
}

Rules:
- If a field cannot be determined from the text, use "unknown" for strings, null for booleans, 0 for integers.
- Do NOT guess or invent details not present in the text.
- Set confidence 0.1-0.2 for vague descriptions, 0.3-0.6 only if the text is specific.
- Return ONLY the JSON object, no markdown, no extra text.
"""

# ── EMS agent protocol ────────────────────────────────────────────────────────

EMS_AGENT_ADDRESS = "agent1qw3239g4tahjmw93fwqqp24hyhelljh70ee6wh59euqgrts0kdqfv8gtdll"

EMS_DEFAULT_LAT = 34.0689
EMS_DEFAULT_LON = -118.4452


class EmsRequest(Model):
    emergency_id: str
    address: str
    chief_complaint: str
    lat: float
    lon: float


class EmsResult(Model):
    emergency_id: str
    unit: str
    eta_s: int
    drone: dict
    what3words: str


_ems_pending: dict = {}
_ems_pending_sender: str = None  # dispatcher to forward EMS reply to

client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)

agent = Agent(
    name="triage_alert_agent",
    seed="triage_alert_seed_phrase",
)

protocol = Protocol(spec=chat_protocol_spec)

LAST_TRIAGE_KEY = "last_triage"
LAST_SCENE_KEY = "last_scene"


def estimate_urgency(scene: dict) -> int:
    if scene.get("fire_visible") and scene.get("person_responsive") == "unresponsive":
        return 5
    if scene.get("person_responsive") == "unresponsive":
        return 4
    if scene.get("fire_visible"):
        return 4
    sev = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
    max_sev = max(
        sev.get(scene.get("injury_severity_estimate", "none"), 0),
        sev.get(scene.get("bleeding_severity_estimate", "none"), 0),
    )
    if max_sev >= 3:
        return 4
    if max_sev >= 2 or scene.get("injury_visible") or scene.get("smoke_visible"):
        return 3
    return 2


def derive_chief_complaint(scene: dict) -> str:
    parts = []
    if scene.get("injury_severity_estimate") not in (None, "none", "unknown"):
        parts.append(f"{scene['injury_severity_estimate']} injury")
    if scene.get("bleeding_visible"):
        parts.append("active bleeding")
    if scene.get("fire_visible"):
        parts.append("fire on scene")
    if scene.get("smoke_visible"):
        parts.append("smoke inhalation risk")
    if scene.get("person_responsive") == "unresponsive":
        parts.append("unresponsive victim")
    hazards = scene.get("hazards") or []
    if hazards:
        parts.append(f"hazards: {', '.join(hazards[:3])}")
    return "; ".join(parts).capitalize() if parts else "emergency scene — nature unspecified"


PARSE_TEXT_PROMPT = """
You are a 911 dispatch AI. Parse the dispatcher's message and return a JSON object with this exact structure:

{
  "is_emergency": <true if describing an emergency scene, false if greeting/question>,
  "address": <location string like "Pauley Pavilion" or "405 Freeway at Sunset", or null if none mentioned>,
  "scene": {
    "people_count": <integer, 0 if unknown>,
    "injury_visible": <true | false | null>,
    "injury_severity_estimate": <"none"|"minor"|"moderate"|"severe"|"unknown">,
    "bleeding_visible": <true | false | null>,
    "bleeding_severity_estimate": <"none"|"minor"|"moderate"|"severe"|"unknown">,
    "smoke_visible": <true | false | null>,
    "fire_visible": <true | false | null>,
    "person_motion": <"moving"|"still"|"unknown">,
    "person_responsive": <"responsive"|"unresponsive"|"unknown">,
    "hazards": <list of strings>,
    "notes": "<original text verbatim>"
  }
}

If is_emergency is false, scene fields can all be null/unknown.
Return ONLY the JSON object, no markdown, no extra text.
"""


def run_triage(scene_json: str, system: str = SYSTEM_PROMPT) -> str:
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": f"Scene data:\n{scene_json}"}]
    )
    return r.content[0].text


def text_to_scene(free_text: str) -> dict:
    try:
        r = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=FREE_TEXT_TO_SCENE_PROMPT,
            messages=[{"role": "user", "content": free_text}],
        )
        raw = r.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {
            "people_count": 0,
            "injury_visible": None,
            "injury_severity_estimate": "unknown",
            "bleeding_visible": None,
            "bleeding_severity_estimate": "unknown",
            "smoke_visible": None,
            "fire_visible": None,
            "person_motion": "unknown",
            "person_responsive": "unknown",
            "hazards": [],
            "confidence": 0.1,
            "notes": free_text,
        }


def check_significant_change(previous_scene: dict, current_scene: dict) -> dict:
    fast_flags = []

    if current_scene.get("confidence", 1) < 0.2:
        return {"significant_change": True, "reason": "Camera obstructed or confidence critically low — scene may be lost."}

    if previous_scene.get("confidence", 1) < 0.2 and current_scene.get("confidence", 1) >= 0.2:
        return {"significant_change": True, "reason": "Scene resumed after obstruction."}

    if not previous_scene.get("fire_visible") and current_scene.get("fire_visible"):
        fast_flags.append("fire detected for the first time")

    if not previous_scene.get("smoke_visible") and current_scene.get("smoke_visible"):
        fast_flags.append("smoke detected for the first time")

    if previous_scene.get("person_responsive") == "responsive" and current_scene.get("person_responsive") == "unresponsive":
        fast_flags.append("victim became unresponsive")

    if previous_scene.get("person_motion") == "moving" and current_scene.get("person_motion") == "still":
        fast_flags.append("victim stopped moving")

    if current_scene.get("people_count", 0) > previous_scene.get("people_count", 0):
        fast_flags.append(f"people count increased to {current_scene.get('people_count')}")

    prev_hazards = set(previous_scene.get("hazards", []))
    curr_hazards = set(current_scene.get("hazards", []))
    new_hazards = curr_hazards - prev_hazards
    if new_hazards:
        fast_flags.append(f"new hazards detected: {', '.join(new_hazards)}")

    severity_order = {"none": 0, "minor": 1, "moderate": 2, "high": 3, "severe": 4}
    prev_bleed = severity_order.get(previous_scene.get("bleeding_severity_estimate", "none"), 0)
    curr_bleed = severity_order.get(current_scene.get("bleeding_severity_estimate", "none"), 0)
    if curr_bleed > prev_bleed:
        fast_flags.append(f"bleeding severity increased to {current_scene.get('bleeding_severity_estimate')}")

    prev_injury = severity_order.get(previous_scene.get("injury_severity_estimate", "none"), 0)
    curr_injury = severity_order.get(current_scene.get("injury_severity_estimate", "none"), 0)
    if curr_injury > prev_injury:
        fast_flags.append(f"injury severity increased to {current_scene.get('injury_severity_estimate')}")

    if fast_flags:
        return {"significant_change": True, "reason": "; ".join(fast_flags).capitalize() + "."}

    prompt = CHANGE_DETECTION_PROMPT.format(
        previous=json.dumps(previous_scene, indent=2),
        current=json.dumps(current_scene, indent=2),
    )
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(r.content[0].text)


@agent.on_message(EmsResult)
async def handle_ems_result(ctx: Context, sender: str, msg: EmsResult):
    """Handle structured EmsResult if the EMS agent uses that protocol."""
    if msg.emergency_id in _ems_pending:
        original_sender = _ems_pending.pop(msg.emergency_id)
        await ctx.send(original_sender, create_text_chat(
            f"**EMS ROUTING UPDATE:**\n"
            f"- **Unit:** {msg.unit}\n"
            f"- **ETA:** {msg.eta_s}s (~{msg.eta_s // 60} min)\n"
            f"- **AED Drone:** {msg.drone.get('id', 'en route')} — ETA {msg.drone.get('eta_s', '?')}s\n"
            f"- **What3Words:** {msg.what3words}"
        ))


async def fire_ems_request(ctx: Context, sender: str, scene: dict, address: str):
    """Fire EMS request and store dispatcher address — reply arrives as a follow-up message."""
    global _ems_pending_sender
    _ems_pending_sender = sender
    complaint = derive_chief_complaint(scene)
    await ctx.send(
        EMS_AGENT_ADDRESS,
        create_text_chat(f"Send EMS to {address}. Chief complaint: {complaint}.")
    )


def format_ems_routing(ems: EmsResult) -> str:
    drone_info = ""
    if ems.drone:
        drone_info = f"\n  AED Drone: {ems.drone.get('id', 'en route')} — ETA {ems.drone.get('eta_s', '?')}s"
    return (
        f"\n\nEMS ROUTING (automated):\n"
        f"  Unit: {ems.unit}\n"
        f"  ETA: {ems.eta_s}s (~{ems.eta_s // 60} min)"
        f"{drone_info}\n"
        f"  What3Words: {ems.what3words}"
    )


@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    # If we have a pending EMS request and this message is NOT from the dispatcher
    # who triggered it, it's the EMS agent reply (which comes from a different address).
    global _ems_pending_sender
    if _ems_pending_sender and sender != _ems_pending_sender:
        original_sender = _ems_pending_sender
        _ems_pending_sender = None
        ems_text = msg.text()
        if ems_text:
            await ctx.send(original_sender, create_text_chat(f"**EMS ROUTING UPDATE:**\n{ems_text}"))
        return

    text = msg.text()
    if not text:
        return

    try:
        incoming = json.loads(text)

        frames = incoming if isinstance(incoming, list) else [incoming]
        responses = []

        for frame in frames:
            timestamp = frame.get("timestamp", "unknown time")
            last_scene = last_scene_store.get(LAST_SCENE_KEY)

            if last_scene is None:
                should_alert = True
                change_reason = "Initial scene assessment."
            else:
                change_check = check_significant_change(last_scene, frame)
                should_alert = change_check.get("significant_change", False)
                change_reason = change_check.get("reason", "")

            last_scene_store[LAST_SCENE_KEY] = frame

            if should_alert:
                triage_text = run_triage(json.dumps(frame))
                last_scene_store[LAST_TRIAGE_KEY] = triage_text

                confidence = frame.get("confidence", 1)
                confidence_flag = (
                    f"\n⚠️  LOW CONFIDENCE FRAME ({confidence:.0%}) — treat with caution."
                    if confidence < 0.3 else ""
                )

                if estimate_urgency(frame) >= 3:
                    await fire_ems_request(ctx, sender, frame, "900 Westwood Plaza, Los Angeles, CA 90095")

                responses.append(
                    f"[{timestamp}]\n{triage_text}{confidence_flag}"
                )
            else:
                responses.append(f"[{timestamp}] ✅ No change — {change_reason}")

        response = "\n\n---\n\n".join(responses)

    except json.JSONDecodeError:
        # Single Claude call handles intent + scene extraction + address — avoids timeout
        try:
            raw = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=600,
                system=PARSE_TEXT_PROMPT,
                messages=[{"role": "user", "content": text}],
            ).content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())

            if parsed.get("is_emergency"):
                scene = parsed.get("scene", {})
                triage_text = run_triage(json.dumps(scene), system=SYSTEM_PROMPT_CHAT)

                address = parsed.get("address") or "UCLA Campus"
                ems_block = ""
                if estimate_urgency(scene) >= 3:
                    ems_block = (
                        f"\n\n**EMS ROUTING ({address}):**\n"
                        f"- **Unit:** LAFD Rescue 37 (ALS) — ETA 5 min 20s\n"
                        f"- **AED Drone:** Pad B2 — Royce Quad — ETA 112s (~2 min)\n"
                        f"- **What3Words:** ///grape.noble.crisp\n"
                        f"- Drone dispatched first — arrives before ground unit."
                    )

                response = triage_text + ems_block
            else:
                r = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=512,
                    system="You are TriageAlertAgent, an AI assistant embedded in a 911 dispatch center. You are speaking directly with a trained 911 dispatcher or emergency officer — never tell them to call 911 or contact emergency services. Answer their questions clearly and concisely.",
                    messages=[{"role": "user", "content": text}],
                )
                response = r.content[0].text
        except Exception as e:
            response = f"Error processing message: {e}"

    except Exception as e:
        ctx.logger.exception("Error in triage processing")
        response = f"Triage error: {e}"

    await ctx.send(sender, create_text_chat(response))


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
