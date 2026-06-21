import asyncio
import json
from datetime import datetime
from uuid import uuid4
from uagents import Agent, Context
from uagents_core.contrib.protocols.chat import (
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    chat_protocol_spec,
)
from uagents import Protocol

# TRIAGE_AGENT_ADDRESS = "agent1q2fu34yahnp7k8t0kyk44kwhn2jlmhe4vl9ljr5z0e93zmh9q5kgz3fmn2v"
TRIAGE_AGENT_ADDRESS = "agent1q09ckygsh9kh4hgk0vpd405e3v6yufujkg4klpm6v26rud4eelhvuazjaz2"

with open("sample_frames50.json") as f:
    frames = json.load(f)

sender = Agent(
    name="test_sender",
    seed="test_sender_seed",
    port=8002,
    endpoint=["http://localhost:8002/submit"],
    network="testnet"
)

protocol = Protocol(spec=chat_protocol_spec)

@sender.on_event("startup")
async def send_frames(ctx: Context):
    for frame in frames:
        msg = ChatMessage(
            timestamp=datetime.utcnow(),
            msg_id=uuid4(),
            content=[TextContent(type="text", text=json.dumps(frame))]
        )
        await ctx.send(TRIAGE_AGENT_ADDRESS, msg)
        ctx.logger.info(f"Sent frame: {frame.get('timestamp')}")
        await asyncio.sleep(2)
    
    ctx.logger.info("All frames sent. Waiting for responses...")
    await asyncio.sleep(30)

@protocol.on_message(ChatMessage)
async def handle_response(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"\n=== TRIAGE RESPONSE ===\n{msg.text()}\n")

@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass

sender.include(protocol)

if __name__ == "__main__":
    sender.run()