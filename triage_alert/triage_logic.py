import os
import re
import json

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
except ImportError:
    pass

import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

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
- silent_distress: whether a silent distress signal was detected (boolean) — highest priority signal; person may be coerced or unable to speak freely
- silent_distress_description: description of what was observed, or null
- audio_distress: whether the caller verbally expressed distress keywords (boolean) — treat as high-priority even without visual confirmation
- audio_keywords: list of distress keywords detected in the caller's speech
- hazards: list of detected hazards (may include "silent_distress", "weapon_visible")
- confidence: confidence score of the analysis (0-1)
- notes: free-text observation notes

Your job is to reason over this data and return a dispatcher-facing report. Prioritize action over data — the dispatcher can already see the raw feed. Lead with what they need to do, not what you observed.

Format your response exactly like this:

[STATUS] e.g. "CRITICAL — IMMEDIATE RESPONSE REQUIRED" or "LOW PRIORITY — NON-URGENT RESPONSE"

DISPATCH:
- List each unit to send with specifics (e.g. "EMS — ALS unit, 2 ambulances minimum")

RESPONDER ACTIONS:
- Numbered, specific steps for arriving responders in order of priority

SCENE SUMMARY:
- Brief bullets of key observations, only what adds context beyond the raw feed

Confidence: [X]% | [timestamp if provided]
⚠️ Visual AI assessment only — confirm with verbal contact.

Urgency scale (use internally to determine STATUS):
1 = Minor, no immediate threat
2 = Low urgency, monitoring needed
3 = Moderate, prompt response required
4 = High urgency, immediate response required
5 = Critical, mass casualty or life-threatening scene

Rules:
- If silent_distress is true, urgency is at least 4; DISPATCH must include law enforcement (possible coercion or domestic violence)
- If audio_distress is true, urgency is at least 3; include the caller's keywords in SCENE SUMMARY
- If confidence is below 0.3, prepend "LOW CONFIDENCE ASSESSMENT —" to the STATUS line
- If fire is visible, always include Fire units
- If person is unresponsive, urgency is at least 4
- If both fire and unresponsive victim are present, urgency is 5
- Only recommend units relevant to the scene
- Never include raw JSON field names in your output
"""

CHANGE_DETECTION_PROMPT = """\
You are comparing two consecutive scene observations from a live 911 emergency feed.

Previous scene:
{previous}

Current scene:
{current}

Determine if the situation has changed in a way that warrants a new alert to the dispatcher.

Flag as a significant change if ANY of the following are true:
- Urgency level changed
- Silent distress newly detected
- Person became unresponsive (was responsive before)
- Person stopped moving (was moving before)
- Fire appeared for the first time
- Smoke appeared for the first time
- New hazards were detected
- Bleeding severity increased
- Injury severity increased
- People count changed
- Camera was covered/obstructed (confidence dropped below 0.2)
- Scene resumed after obstruction

Do NOT flag as significant if:
- Scene is nearly identical with only minor confidence fluctuation
- Same hazards, same victim status, same urgency
- Situation appears to have improved or de-escalated — dispatch has already been alerted and does not need a new alert for de-escalation; only escalation warrants a new dispatch alert

Respond with only a JSON object, no extra text:
{{
  "significant_change": true or false,
  "reason": "<one sentence explaining what changed, or why it did not change>"
}}"""

_SEV          = {"low": 1, "minor": 1, "moderate": 2, "high": 3, "severe": 4}
_SEV_TO_TRIAGE = {"low": "minor", "moderate": "moderate", "high": "severe"}


def adapt_frame(rich: dict) -> dict:
    """Convert analyzeFrame.js rich schema to the flat triage schema."""
    people   = rich.get("people") or []
    injuries = rich.get("injuries") or []

    people_count      = len(people)
    motions           = [p.get("motion") for p in people]
    responsives       = [p.get("responsive") for p in people]
    person_motion     = "moving" if "moving" in motions else ("still" if people else "unknown")
    person_responsive = "unresponsive" if "unresponsive" in responsives else ("responsive" if people else "unknown")

    if injuries:
        worst                    = max(injuries, key=lambda i: _SEV.get(i.get("severity", "low"), 0))
        injury_visible           = True
        injury_severity_estimate = _SEV_TO_TRIAGE.get(worst.get("severity", "low"), "minor")
        injury_location          = worst.get("body_part")
        bleeding_visible         = any(i.get("bleeding") for i in injuries)
        bleeders                 = [i for i in injuries if i.get("bleeding")]
        bleeding_severity_estimate = (
            _SEV_TO_TRIAGE.get(
                max(bleeders, key=lambda i: _SEV.get(i.get("severity", "low"), 0)).get("severity", "low"),
                "minor",
            )
            if bleeders else "none"
        )
    else:
        injury_visible = bleeding_visible = False
        injury_severity_estimate = bleeding_severity_estimate = "none"
        injury_location = None

    hazards = list(rich.get("hazards") or [])
    if rich.get("silent_distress") and "silent_distress" not in hazards:
        hazards.insert(0, "silent_distress")
    if rich.get("audio_distress") and "audio_distress" not in hazards:
        hazards.insert(0, "audio_distress")
    if (rich.get("objects") or {}).get("weapons_visible") and "weapon_visible" not in hazards:
        hazards.append("weapon_visible")

    notes_parts = []
    if rich.get("silent_distress") and rich.get("silent_distress_description"):
        notes_parts.append(f"SILENT DISTRESS: {rich['silent_distress_description']}")
    if rich.get("audio_distress") and rich.get("audio_keywords"):
        notes_parts.append(f"AUDIO DISTRESS — caller said distress keywords: {', '.join(rich['audio_keywords'][:5])}")
    if rich.get("notes"):
        notes_parts.append(rich["notes"])

    return {
        "people_count":                people_count,
        "injury_visible":              injury_visible,
        "injury_severity_estimate":    injury_severity_estimate,
        "injury_location":             injury_location,
        "bleeding_visible":            bleeding_visible,
        "bleeding_severity_estimate":  bleeding_severity_estimate,
        "smoke_visible":               rich.get("smoke_visible", False),
        "fire_visible":                rich.get("fire_visible", False),
        "person_motion":               person_motion,
        "person_responsive":           person_responsive,
        "hazards":                     hazards,
        "silent_distress":             rich.get("silent_distress", False),
        "silent_distress_description": rich.get("silent_distress_description"),
        "audio_distress":              rich.get("audio_distress", False),
        "audio_keywords":              rich.get("audio_keywords") or [],
        "frame_quality":               rich.get("frame_quality", "unknown"),
        "confidence":                  rich.get("confidence", 0),
        "notes":                       " | ".join(notes_parts),
        "timestamp":                   rich.get("timestamp"),
    }


def run_triage(scene_json: str) -> str:
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Scene data:\n{scene_json}"}],
    )
    return r.content[0].text


def check_significant_change(previous_scene: dict, current_scene: dict) -> dict:
    fast_flags = []

    if current_scene.get("confidence", 1) < 0.2:
        return {"significant_change": True, "reason": "Camera obstructed or confidence critically low — scene may be lost."}

    if previous_scene.get("confidence", 1) < 0.2 and current_scene.get("confidence", 1) >= 0.2:
        return {"significant_change": True, "reason": "Scene resumed after obstruction."}

    if not previous_scene.get("silent_distress") and current_scene.get("silent_distress"):
        desc = current_scene.get("silent_distress_description") or "visual signal"
        fast_flags.append(f"silent distress detected: {desc}")

    if not previous_scene.get("audio_distress") and current_scene.get("audio_distress"):
        kws = current_scene.get("audio_keywords") or []
        fast_flags.append(f"caller audio distress: {', '.join(kws[:4]) if kws else 'distress keywords detected'}")

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

    prev_hazards = set(previous_scene.get("hazards") or [])
    curr_hazards = set(current_scene.get("hazards") or [])
    new_hazards  = curr_hazards - prev_hazards
    if new_hazards:
        fast_flags.append(f"new hazards detected: {', '.join(sorted(new_hazards))}")

    severity_order = {"none": 0, "minor": 1, "moderate": 2, "high": 3, "severe": 4}
    if severity_order.get(current_scene.get("bleeding_severity_estimate", "none"), 0) > \
       severity_order.get(previous_scene.get("bleeding_severity_estimate", "none"), 0):
        fast_flags.append(f"bleeding severity increased to {current_scene.get('bleeding_severity_estimate')}")

    if severity_order.get(current_scene.get("injury_severity_estimate", "none"), 0) > \
       severity_order.get(previous_scene.get("injury_severity_estimate", "none"), 0):
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
        messages=[{"role": "user", "content": prompt}],
    )
    raw = r.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            return json.loads(m.group(1).strip())
        raise
