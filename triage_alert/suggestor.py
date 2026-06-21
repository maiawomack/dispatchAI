# from datetime import datetime
# from uuid import uuid4
# import json

# import anthropic
# from uagents import Context, Protocol, Agent
# from uagents_core.contrib.protocols.chat import (
#     ChatAcknowledgement,
#     ChatMessage,
#     EndSessionContent,
#     StartSessionContent,
#     TextContent,
#     chat_protocol_spec,
# )

# def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
#     content = [TextContent(type="text", text=text)]
#     if end_session:
#         content.append(EndSessionContent(type="end-session"))
#     return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=content)


# SYSTEM_PROMPT = """
# You are TriageAlertAgent, a medical and emergency triage AI embedded in a live 911 dispatch system.

# You receive structured JSON from a VisionAgent describing a live emergency scene. The JSON contains:
# - people_count: number of visible people
# - injury_visible: whether any injuries are visible (boolean)
# - injury_severity_estimate: estimated severity ("none", "minor", "moderate", "severe")
# - injury_location: body part injured, or null
# - bleeding_visible: whether bleeding is visible (boolean)
# - bleeding_severity_estimate: estimated bleeding severity ("none", "minor", "moderate", "severe")
# - smoke_visible: whether smoke is visible (boolean)
# - fire_visible: whether fire is visible (boolean)
# - person_motion: whether the person is "moving" or "still"
# - person_responsive: whether the person is "responsive" or "unresponsive"
# - hazards: list of detected hazards
# - confidence: confidence score of the analysis (0-1)
# - notes: free-text observation notes

# Your job is to reason over this data and return a triage decision in the following JSON format:
# {
#   "urgency_level": <1-5>,
#   "recommended_units": ["EMS", "Fire", "Police"],
#   "summary": "<plain-language summary a dispatcher can read at a glance>"
# }

# Urgency scale:
# 1 = Minor, no immediate threat
# 2 = Low urgency, monitoring needed
# 3 = Moderate, prompt response required
# 4 = High urgency, immediate response required
# 5 = Critical, mass casualty or life-threatening scene

# Only recommend units relevant to the scene. Always return valid JSON with no extra text.
# """

# CHANGE_DETECTION_PROMPT = """
# You are comparing two triage decisions to determine if the scene has changed significantly.

# Previous triage:
# {previous}

# Current triage:
# {current}

# A significant change is one of:
# - Urgency level increased
# - New victim(s) detected
# - New hazard or fire appeared
# - Recommended units changed
# - Person became unresponsive or stopped moving

# Respond with only a JSON object:
# {{
#   "significant_change": true or false,
#   "reason": "<brief explanation>"
# }}
# """

# client = anthropic.Anthropic(
#     api_key="INSERT_YOUR_CLAUDE_API_KEY_HERE"
# )

# agent = Agent(
#     name="triage_alert_agent",
#     seed="triage_alert_seed_phrase",
#     port=8001,
#     endpoint=["http://localhost:8001/submit"],
# )

# protocol = Protocol(spec=chat_protocol_spec)

# LAST_TRIAGE_KEY = "last_triage"


# def run_triage(scene_json: str) -> dict:
#     r = client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=512,
#         system=SYSTEM_PROMPT,
#         messages=[{"role": "user", "content": f"Scene data:\n{scene_json}"}]
#     )
#     return json.loads(r.content[0].text)


# def check_significant_change(previous: dict, current: dict) -> dict:
#     prompt = CHANGE_DETECTION_PROMPT.format(
#         previous=json.dumps(previous, indent=2),
#         current=json.dumps(current, indent=2),
#     )
#     r = client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=256,
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return json.loads(r.content[0].text)


# @protocol.on_message(ChatMessage)
# async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
#     await ctx.send(
#         sender,
#         ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
#     )

#     text = msg.text()
#     if not text:
#         return

#     try:
#         scene_data = json.loads(text)
#         triage_result = run_triage(json.dumps(scene_data))
#         last_triage = ctx.storage.get(LAST_TRIAGE_KEY)

#         if last_triage is None:
#             should_alert = True
#             change_reason = "Initial scene assessment."
#         else:
#             change_check = check_significant_change(last_triage, triage_result)
#             should_alert = change_check.get("significant_change", False)
#             change_reason = change_check.get("reason", "")

#         ctx.storage.set(LAST_TRIAGE_KEY, triage_result)

#         if should_alert:
#             urgency = triage_result.get("urgency_level", "?")
#             units = ", ".join(triage_result.get("recommended_units", []))
#             summary = triage_result.get("summary", "No summary available.")
#             response = (
#                 f"TRIAGE ALERT\n"
#                 f"Urgency Level: {urgency}/5\n"
#                 f"Dispatch: {units}\n"
#                 f"Summary: {summary}\n"
#                 f"Change detected: {change_reason}"
#             )
#         else:
#             response = (
#                 f"No significant change detected. Scene stable.\n"
#                 f"Reason: {change_reason}\n"
#                 f"Current urgency: {triage_result.get('urgency_level')}/5"
#             )

#     except json.JSONDecodeError:
#         try:
#             r = client.messages.create(
#                 model="claude-sonnet-4-6",
#                 max_tokens=512,
#                 system="You are a helpful emergency triage assistant. Answer dispatcher questions clearly and concisely.",
#                 messages=[{"role": "user", "content": text}]
#             )
#             response = r.content[0].text
#         except Exception as e:
#             response = f"Error processing question: {e}"

#     except Exception as e:
#         ctx.logger.exception("Error in triage processing")
#         response = f"Triage error: {e}"

#     await ctx.send(sender, create_text_chat(response))


# @protocol.on_message(ChatAcknowledgement)
# async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
#     pass


# agent.include(protocol, publish_manifest=True)

# if __name__ == "__main__":
#     agent.run()


from datetime import datetime
from uuid import uuid4
import json

import anthropic
from uagents import Context, Protocol, Agent
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(timestamp=datetime.utcnow(), msg_id=uuid4(), content=content)


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

Your job is to reason over this data and return a triage decision in the following JSON format:
{
  "urgency_level": <1-5>,
  "recommended_units": ["EMS", "Fire", "Police"],
  "summary": "<plain-language summary a dispatcher can read at a glance>",
  "key_concerns": ["<list of the most critical observations driving this decision>"]
}

Urgency scale:
1 = Minor, no immediate threat
2 = Low urgency, monitoring needed
3 = Moderate, prompt response required
4 = High urgency, immediate response required
5 = Critical, mass casualty or life-threatening scene

Rules:
- If confidence is below 0.3, flag the frame as LOW CONFIDENCE and note it in the summary but still assess based on available data
- If fire is visible, always include Fire units
- If person is unresponsive, urgency is at least 4
- If both fire and unresponsive victim are present, urgency is 5
- Only recommend units relevant to the scene
- Always return valid JSON with no extra text
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

client = anthropic.Anthropic(
    api_key=""
)

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


def run_triage(scene_json: str) -> dict:
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Scene data:\n{scene_json}"}]
    )
    return json.loads(r.content[0].text)


def check_significant_change(previous_scene: dict, current_scene: dict) -> dict:
    """Compare raw scene data directly for more accurate change detection."""

    # Hard-coded fast checks before even calling the LLM
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

    # If no fast flags, let the LLM make the final call
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


@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    text = msg.text()
    if not text:
        return

    try:
        incoming = json.loads(text)

        # Accept either a single frame or a list of frames
        frames = incoming if isinstance(incoming, list) else [incoming]

        responses = []

        for frame in frames:
            timestamp = frame.get("timestamp", "unknown time")
            last_scene = ctx.storage.get(LAST_SCENE_KEY)

            if last_scene is None:
                # First frame — always triage
                should_alert = True
                change_reason = "Initial scene assessment."
            else:
                change_check = check_significant_change(last_scene, frame)
                should_alert = change_check.get("significant_change", False)
                change_reason = change_check.get("reason", "")

            ctx.storage.set(LAST_SCENE_KEY, frame)

            if should_alert:
                triage_result = run_triage(json.dumps(frame))
                ctx.storage.set(LAST_TRIAGE_KEY, triage_result)

                urgency = triage_result.get("urgency_level", "?")
                units = ", ".join(triage_result.get("recommended_units", []))
                summary = triage_result.get("summary", "No summary available.")
                concerns = triage_result.get("key_concerns", [])
                concerns_str = "\n  - " + "\n  - ".join(concerns) if concerns else ""

                confidence = frame.get("confidence", 1)
                confidence_flag = f"\n⚠️  LOW CONFIDENCE FRAME ({confidence:.0%}) — treat with caution." if confidence < 0.3 else ""

                responses.append(
                    f"[{timestamp}]\n"
                    f"🚨 TRIAGE ALERT\n"
                    f"Urgency Level: {urgency}/5\n"
                    f"Dispatch: {units}\n"
                    f"Summary: {summary}\n"
                    f"Key concerns:{concerns_str}\n"
                    f"Trigger: {change_reason}"
                    f"{confidence_flag}"
                )
            else:
                responses.append(
                    f"[{timestamp}] ✅ No change — {change_reason}"
                )

        response = "\n\n---\n\n".join(responses)

    except json.JSONDecodeError:
        try:
            r = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=512,
                system="You are a helpful emergency triage assistant. Answer dispatcher questions clearly and concisely.",
                messages=[{"role": "user", "content": text}]
            )
            response = r.content[0].text
        except Exception as e:
            response = f"Error processing question: {e}"

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