from datetime import datetime
from uuid import uuid4
from uagents import Agent, Context
from uagents import Agent, Context, Model

class ChatMessage(Model):
    text: str

agent = Agent(
    name="Ananya",
    seed="secret_seed_phrase",
    port=8000,
    endpoint=["http://localhost:8000/submit"],
)

@agent.on_event("startup")
async def startup_function(ctx: Context):
    ctx.logger.info(f"Dispatch AI agent started. Address: {agent.address}")

@agent.on_message(model=ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"Message from {sender}: {msg.text}")
    await ctx.send(sender, ChatMessage(
        text=(
            "🚨 Dispatch AI activated. A video link has been sent to the caller. "
            "Our vision agent is analyzing the scene and will produce a dispatch brief shortly."
        )
    ))

if __name__ == "__main__":
    agent.run()