from datetime import datetime
from uuid import uuid4
import json

import anthropic
from uagents import Context, Protocol, Agent
from uagents.experimental.chat_agent.protocol import build_llm_message_history
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
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
- victim_count: number of visible victims
- visible_injuries: list of observed injuries (e.g. "laceration", "burns", "unconscious")
- detected_hazards: list of hazards (e.g. "smoke", "downed power line", "gas leak")
- fire_presence: boolean

Your job is to reason over this data and return a triage decision in the following JSON format:
{
  "urgency_level": <1-5>,
  "recommended_units": ["EMS", "Fire", "Police"],
  "summary": "<plain-language summary a dispatcher can read at a glance>"
}

Urgency scale:
1 = Minor, no immediate threat
2 = Low urgency, monitoring needed
3 = Moderate, prompt response required
4 = High urgency, immediate response required
5 = Critical, mass casualty or life-threatening scene

Only recommend units relevant to the scene. Always return valid JSON with no extra text.
"""

CHANGE_DETECTION_PROMPT = """
You are comparing two triage decisions to determine if the scene has changed significantly.

Previous triage:
{previous}

Current triage:
{current}

A significant change is one of:
- Urgency level increased
- New victim(s) detected
- New hazard or fire appeared
- Recommended units changed

Respond with only a JSON object:
{
  "significant_change": true or false,
  "reason": "<brief explanation>"
}
"""

client = anthropic.Anthropic(
    api_key="INSERT_YOUR_CLAUDE_API_KEY_HERE"
)

agent = Agent(
    name="triage_alert_agent",
    seed="triage_alert_seed_phrase",
    port=8001,
    endpoint=["http://localhost:8001/submit"],
)

protocol = Protocol(spec=chat_protocol_spec)

# Store the last triage result in agent storage to detect changes across messages
LAST_TRIAGE_KEY = "last_triage"


def run_triage(scene_json: str) -> dict:
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Scene data:\n{scene_json}"}]
    )
    return json.loads(r.content[0].text)


def check_significant_change(previous: dict, current: dict) -> dict:
    prompt = CHANGE_DETECTION_PROMPT.format(
        previous=json.dumps(previous, indent=2),
        current=json.dumps(current, indent=2),
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
        # Attempt to parse input as scene JSON from VisionAgent
        scene_data = json.loads(text)

        # Step 1: Run triage reasoning
        triage_result = run_triage(json.dumps(scene_data))

        # Step 2: Compare to previous triage output
        last_triage = ctx.storage.get(LAST_TRIAGE_KEY)

        if last_triage is None:
            # First reading — always alert
            should_alert = True
            change_reason = "Initial scene assessment."
        else:
            change_check = check_significant_change(last_triage, triage_result)
            should_alert = change_check.get("significant_change", False)
            change_reason = change_check.get("reason", "")

        # Step 3: Save current triage as the new baseline
        ctx.storage.set(LAST_TRIAGE_KEY, triage_result)

        # Step 4: Build dispatcher-facing response
        if should_alert:
            urgency = triage_result.get("urgency_level", "?")
            units = ", ".join(triage_result.get("recommended_units", []))
            summary = triage_result.get("summary", "No summary available.")

            response = (
                f"🚨 TRIAGE ALERT\n"
                f"Urgency Level: {urgency}/5\n"
                f"Dispatch: {units}\n"
                f"Summary: {summary}\n"
                f"Change detected: {change_reason}"
            )
        else:
            response = (
                f"✅ No significant change detected. Scene stable.\n"
                f"Reason: {change_reason}\n"
                f"Current urgency: {triage_result.get('urgency_level')}/5"
            )

    except json.JSONDecodeError:
        # Input wasn't JSON — treat as a plain dispatcher question
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