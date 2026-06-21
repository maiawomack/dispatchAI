from datetime import datetime
from uuid import uuid4
import asyncio
import json
import os
import sys
import atexit

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
except ImportError:
    pass

import anthropic
from uagents import Context, Protocol, Agent, Model
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

from arize.otel import register
from opentelemetry import trace
from opentelemetry.trace import StatusCode

# ── Credentials — fail fast if any required env var is missing ───────────────
_ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_ARIZE_SPACE_ID    = os.environ.get("ARIZE_SPACE_ID", "")
_ARIZE_API_KEY     = os.environ.get("ARIZE_API_KEY", "")

_missing = [name for name, val in [
    ("ANTHROPIC_API_KEY", _ANTHROPIC_API_KEY),
    ("ARIZE_SPACE_ID",    _ARIZE_SPACE_ID),
    ("ARIZE_API_KEY",     _ARIZE_API_KEY),
] if not val]
if _missing:
    print(f"[triage_alert] FATAL — missing required env vars: {', '.join(_missing)}", file=sys.stderr)
    print("Set them in triage_alert/../.env or export them before starting.", file=sys.stderr)
    sys.exit(1)

# ── Arize tracing setup ──────────────────────────────────────────────────────
tracer_provider = register(
    space_id=_ARIZE_SPACE_ID,
    api_key=_ARIZE_API_KEY,
    project_name="dispatch-ai-triage-agent",
)
atexit.register(tracer_provider.force_flush)

from openinference.instrumentation.anthropic import AnthropicInstrumentor
AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

tracer = trace.get_tracer("triage_alert_agent")

# ─────────────────────────────────────────────────────────────────────────────

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

# Used to parse dispatcher free text into the structured scene format that
# run_triage() expects. Confidence is deliberately capped low because this
# is inferred from words, not observed sensor data.
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
  "confidence": <float 0.1–0.6; keep low since this is inferred not observed>,
  "notes": "<original dispatcher text verbatim>"
}

Rules:
- If a field cannot be determined from the text, use "unknown" for strings, null for booleans, 0 for integers.
- Do NOT guess or invent details not present in the text.
- Set confidence 0.1–0.2 for vague descriptions, 0.3–0.6 only if the text is specific.
- Return ONLY the JSON object, no markdown, no extra text.
"""

# ── EMS agent protocol ────────────────────────────────────────────────────────

EMS_AGENT_ADDRESS = "agent1qw3239g4tahjmw93fwqqp24hyhelljh70ee6wh59euqgrts0kdqfv8gtdll"

# Placeholder coordinates near UCLA campus — replace with real GPS when available
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


# Pending EMS requests: emergency_id -> (asyncio.Event, result_holder)
_ems_pending: dict = {}

client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)

agent = Agent(
    name="triage_alert_agent",
    seed="triage_alert_seed_phrase",
    port=8001,
    endpoint=["http://localhost:8001/submit"],
    network="testnet"
)

protocol = Protocol(spec=chat_protocol_spec)

LAST_TRIAGE_KEY = "last_triage"
LAST_SCENE_KEY = "last_scene"


def estimate_urgency(scene: dict) -> int:
    """Derive a 1–5 urgency level from raw scene fields (mirrors SYSTEM_PROMPT rules)."""
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


def run_triage(scene_json: str) -> str:
    """Run triage LLM call, traced as a child span with scene metadata."""
    with tracer.start_as_current_span("triage.run_triage") as span:
        try:
            scene_data = json.loads(scene_json)
            span.set_attribute("scene.people_count", scene_data.get("people_count", 0))
            span.set_attribute("scene.fire_visible", bool(scene_data.get("fire_visible")))
            span.set_attribute("scene.smoke_visible", bool(scene_data.get("smoke_visible")))
            span.set_attribute("scene.person_responsive", scene_data.get("person_responsive", "unknown"))
            span.set_attribute("scene.injury_severity", scene_data.get("injury_severity_estimate", "none"))
            span.set_attribute("scene.bleeding_severity", scene_data.get("bleeding_severity_estimate", "none"))
            span.set_attribute("scene.confidence", scene_data.get("confidence", 1.0))
            span.set_attribute("llm.model", "claude-sonnet-4-6")
            span.set_attribute("llm.task", "triage_decision")

            # AnthropicInstrumentor auto-traces this call; the outer span adds business context
            r = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Scene data:\n{scene_json}"}]
            )
            result = r.content[0].text
            span.set_attribute("triage.output", result[:500])
            span.set_status(StatusCode.OK)
            return result

        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise


def text_to_scene(free_text: str) -> dict:
    """Convert dispatcher free text to a structured scene dict, then pass to run_triage()."""
    with tracer.start_as_current_span("triage.text_to_scene") as span:
        span.set_attribute("llm.model", "claude-sonnet-4-6")
        span.set_attribute("llm.task", "free_text_to_scene")
        try:
            r = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system=FREE_TEXT_TO_SCENE_PROMPT,
                messages=[{"role": "user", "content": free_text}],
            )
            raw = r.content[0].text.strip()
            # Strip markdown fences if the model wrapped the JSON
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            scene = json.loads(raw.strip())
            span.set_attribute("scene.confidence", scene.get("confidence", 0))
            span.set_status(StatusCode.OK)
            return scene
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            # Return a minimal unknown scene so run_triage() still executes
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
    """Compare raw scene data; fast-path heuristics first, LLM fallback second.
    The outer span captures the final decision so Arize evaluators can judge it.
    """
    with tracer.start_as_current_span("triage.change_detection") as span:
        span.set_attribute("scene.confidence_current", current_scene.get("confidence", 1.0))
        span.set_attribute("scene.confidence_previous", previous_scene.get("confidence", 1.0))

        try:
            # ── Fast heuristic checks ─────────────────────────────────────────
            with tracer.start_as_current_span("triage.change_detection.fast_checks") as fast_span:
                fast_flags = []

                if current_scene.get("confidence", 1) < 0.2:
                    result = {
                        "significant_change": True,
                        "reason": "Camera obstructed or confidence critically low — scene may be lost.",
                    }
                    fast_span.set_attribute("change.triggered_by", "low_confidence_obstruction")
                    _record_change_result(span, result, method="fast_check")
                    return result

                if previous_scene.get("confidence", 1) < 0.2 and current_scene.get("confidence", 1) >= 0.2:
                    result = {"significant_change": True, "reason": "Scene resumed after obstruction."}
                    fast_span.set_attribute("change.triggered_by", "scene_resumed")
                    _record_change_result(span, result, method="fast_check")
                    return result

                if not previous_scene.get("fire_visible") and current_scene.get("fire_visible"):
                    fast_flags.append("fire detected for the first time")

                if not previous_scene.get("smoke_visible") and current_scene.get("smoke_visible"):
                    fast_flags.append("smoke detected for the first time")

                if (previous_scene.get("person_responsive") == "responsive"
                        and current_scene.get("person_responsive") == "unresponsive"):
                    fast_flags.append("victim became unresponsive")

                if (previous_scene.get("person_motion") == "moving"
                        and current_scene.get("person_motion") == "still"):
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

                fast_span.set_attribute("change.fast_flags_count", len(fast_flags))
                fast_span.set_attribute("change.fast_flags", json.dumps(fast_flags))

                if fast_flags:
                    result = {
                        "significant_change": True,
                        "reason": "; ".join(fast_flags).capitalize() + ".",
                    }
                    _record_change_result(span, result, method="fast_check")
                    return result

            # ── LLM fallback ──────────────────────────────────────────────────
            with tracer.start_as_current_span("triage.change_detection.llm_fallback") as llm_span:
                llm_span.set_attribute("llm.model", "claude-sonnet-4-6")
                llm_span.set_attribute("llm.task", "change_detection")

                prompt = CHANGE_DETECTION_PROMPT.format(
                    previous=json.dumps(previous_scene, indent=2),
                    current=json.dumps(current_scene, indent=2),
                )
                r = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = json.loads(r.content[0].text)
                _record_change_result(span, result, method="llm_fallback")
                return result

        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise


def _record_change_result(span, result: dict, method: str):
    """Stamp the final change-detection decision onto the parent span."""
    span.set_attribute("change.significant", result.get("significant_change", False))
    span.set_attribute("change.reason", result.get("reason", ""))
    span.set_attribute("change.detection_method", method)
    span.set_status(StatusCode.OK)


@agent.on_message(EmsResult)
async def handle_ems_result(ctx: Context, sender: str, msg: EmsResult):
    """Receive EmsResult from the EMS & AED Drone agent and unblock the waiting handler."""
    if msg.emergency_id in _ems_pending:
        event, holder = _ems_pending[msg.emergency_id]
        holder["result"] = msg
        event.set()


async def request_ems_routing(ctx: Context, scene: dict):
    """Send EmsRequest and wait up to 8 s; returns EmsResult or None on timeout."""
    ems_id = str(uuid4())
    event = asyncio.Event()
    holder: dict = {}
    _ems_pending[ems_id] = (event, holder)
    try:
        await ctx.send(
            EMS_AGENT_ADDRESS,
            EmsRequest(
                emergency_id=ems_id,
                address="900 Westwood Plaza, Los Angeles, CA 90095",
                chief_complaint=derive_chief_complaint(scene),
                lat=EMS_DEFAULT_LAT,
                lon=EMS_DEFAULT_LON,
            ),
        )
        await asyncio.wait_for(event.wait(), timeout=8.0)
        return holder.get("result")
    except asyncio.TimeoutError:
        return None
    finally:
        _ems_pending.pop(ems_id, None)


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
    """Top-level message handler — each inbound message gets its own root span."""
    with tracer.start_as_current_span("triage.handle_message") as root_span:
        root_span.set_attribute("agent.sender", sender)

        await ctx.send(
            sender,
            ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
        )

        text = msg.text()
        if not text:
            root_span.set_attribute("message.empty", True)
            return

        try:
            incoming = json.loads(text)

            # Accept either a single frame or a list of frames
            frames = incoming if isinstance(incoming, list) else [incoming]
            root_span.set_attribute("frames.count", len(frames))

            responses = []

            for frame_idx, frame in enumerate(frames):
                with tracer.start_as_current_span(f"triage.process_frame.{frame_idx}") as frame_span:
                    timestamp = frame.get("timestamp", "unknown time")
                    frame_span.set_attribute("frame.timestamp", str(timestamp))
                    frame_span.set_attribute("frame.confidence", frame.get("confidence", 1.0))

                    last_scene = last_scene_store.get(LAST_SCENE_KEY)

                    if last_scene is None:
                        should_alert = True
                        change_reason = "Initial scene assessment."
                        frame_span.set_attribute("frame.is_initial", True)
                    else:
                        change_check = check_significant_change(last_scene, frame)
                        should_alert = change_check.get("significant_change", False)
                        change_reason = change_check.get("reason", "")
                        frame_span.set_attribute("frame.is_initial", False)

                    frame_span.set_attribute("frame.should_alert", should_alert)
                    frame_span.set_attribute("frame.change_reason", change_reason)
                    last_scene_store[LAST_SCENE_KEY] = frame

                    if should_alert:
                        triage_text = run_triage(json.dumps(frame))
                        last_scene_store[LAST_TRIAGE_KEY] = triage_text

                        confidence = frame.get("confidence", 1)
                        confidence_flag = (
                            f"\n⚠️  LOW CONFIDENCE FRAME ({confidence:.0%}) — treat with caution."
                            if confidence < 0.3 else ""
                        )

                        root_span.set_attribute("triage.alerted", True)
                        root_span.set_attribute("triage.output_preview", triage_text[:200])

                        ems_block = ""
                        urgency = estimate_urgency(frame)
                        if urgency >= 3:
                            ems_result = await request_ems_routing(ctx, frame)
                            if ems_result:
                                ems_block = format_ems_routing(ems_result)
                            else:
                                ems_block = "\n\n⚠️ EMS routing data unavailable — dispatch manually through CAD."

                        responses.append(
                            f"[{timestamp}]\n"
                            f"{triage_text}"
                            f"{confidence_flag}"
                            f"{ems_block}"
                        )
                    else:
                        root_span.set_attribute("triage.alerted", False)
                        responses.append(f"[{timestamp}] ✅ No change — {change_reason}")

            response = "\n\n---\n\n".join(responses)
            root_span.set_status(StatusCode.OK)

        except json.JSONDecodeError:
            # Dispatcher typed plain text — convert to structured scene JSON, then
            # run the same triage pipeline (including EMS routing) as the JSON path.
            root_span.set_attribute("message.type", "natural_language_query")
            try:
                scene = text_to_scene(text)
                triage_text = run_triage(json.dumps(scene))

                ems_block = ""
                urgency = estimate_urgency(scene)
                if urgency >= 3:
                    ems_result = await request_ems_routing(ctx, scene)
                    if ems_result:
                        ems_block = format_ems_routing(ems_result)
                    else:
                        ems_block = "\n\n⚠️ EMS routing data unavailable — dispatch manually through CAD."

                response = triage_text + ems_block
                root_span.set_status(StatusCode.OK)
            except Exception as e:
                root_span.set_status(StatusCode.ERROR, str(e))
                root_span.record_exception(e)
                response = f"Error processing situation: {e}"

        except Exception as e:
            ctx.logger.exception("Error in triage processing")
            root_span.set_status(StatusCode.ERROR, str(e))
            root_span.record_exception(e)
            response = f"Triage error: {e}"

    await ctx.send(sender, create_text_chat(response))


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
