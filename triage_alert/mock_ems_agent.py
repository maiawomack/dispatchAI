from datetime import datetime
from uuid import uuid4
import random

from uagents import Context, Protocol, Agent
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    chat_protocol_spec,
)

agent = Agent(name="mock_ems_agent")
protocol = Protocol(spec=chat_protocol_spec)

UNITS = ["LAFD Rescue 37 (ALS)", "LAFD Rescue 92 (ALS)", "LAFD Rescue 19 (ALS)"]
PADS  = ["A1 — Lot 4 North", "B2 — Royce Quad", "C3 — Powell Library"]
W3W   = ["sofa.tiger.lamp", "grape.noble.crisp", "army.trend.clock"]


def create_text_chat(text: str) -> ChatMessage:
    return ChatMessage(
        timestamp=datetime.utcnow(),
        msg_id=uuid4(),
        content=[TextContent(type="text", text=text)],
    )


@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(), acknowledged_msg_id=msg.msg_id),
    )

    unit    = random.choice(UNITS)
    pad     = random.choice(PADS)
    w3w     = random.choice(W3W)
    eta_g   = random.randint(240, 420)   # ground ETA 4–7 min
    eta_d   = random.randint(90, 180)    # drone ETA 1.5–3 min

    response = (
        f"**Unit:** {unit} — ETA {eta_g // 60} min {eta_g % 60}s\n"
        f"**AED Drone:** Pad {pad} — ETA {eta_d}s (~{eta_d // 60} min)\n"
        f"**What3Words:** ///{w3w}\n"
        f"Drone dispatched first — arrives before ground unit."
    )

    await ctx.send(sender, create_text_chat(response))


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass


agent.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
