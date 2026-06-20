from uagents import Agent, Context

# instantiate agent
agent = Agent(
    name="Ananya",
    seed="secret_seed_phrase",
    port=8000,
    handle_messages_concurrently=True,
    endpoint=["http://localhost:8000/submit"],
)

# startup handler
@agent.on_event("startup")
async def startup_function(ctx: Context):
    ctx.logger.info(f"Hello, I'm agent {agent.name} and my address is {agent.address}.")

if __name__ == "__main__":
    agent.run()

# Agent parameters:

# name: Identifies the agent (here, “alice”).
# seed: Sets a deterministic seed, generating fixed addresses each time.
# port and endpoint: Configure where the agent will be available.
# Behavior on startup:

# The @agent.on_event("startup") decorator sets a function that runs as soon
# as the agent launches. In this sample, the agent logs a message including 
# its name and unique address.