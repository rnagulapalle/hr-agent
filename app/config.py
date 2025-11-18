import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "hr-workflow-agent")
