# app/agent_react.py

import json
from typing import Dict, Any, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from .config import ANTHROPIC_MODEL
from .tools import (
    get_pto_balance,
    update_emergency_contact,
    create_hr_ticket,
    search_hr_policies,
    create_hardware_request,
)

# Map tool names -> actual tool objects
TOOLS: Dict[str, Any] = {
    "get_pto_balance": get_pto_balance,
    "search_hr_policies": search_hr_policies,
    "update_emergency_contact": update_emergency_contact,
    "create_hr_ticket": create_hr_ticket,
    "create_hardware_request": create_hardware_request,
}

TOOL_DESCRIPTIONS = """
You can use these tools:

1. get_pto_balance
   - description: Get the PTO balance for a user.
   - args: user_id (string)

2. search_hr_policies
   - description: Search HR policy docs.
   - args: query (string)

3. update_emergency_contact
   - description: Update emergency contact info.
   - args:
       user_id (string),
       name (string),
       phone (string),
       relationship (string)

4. create_hr_ticket
   - description: Create an HR ticket if automation is not enough.
   - args:
       user_id (string),
       summary (string),
       details (string)

5. create_hardware_request
   - description: Create a hardware or equipment request (e.g., laptop, monitor).
   - args:
       user_id (string),
       item (string),
       justification (string)
"""

SYSTEM_PROMPT = """
You are an HR workflow assistant for employees.

You have access to tools to:
- get PTO balances
- search HR policies
- update emergency contacts
- create hardware requests
- create HR tickets

You will work in a THINK → ACT → OBSERVE loop.

Output format rules:

- If you need to call a tool, respond with a JSON object with keys:
  - "type": the string "tool_call"
  - "tool": the tool name (e.g. "get_pto_balance")
  - "arguments": an object containing the arguments for that tool

- If you are ready to answer the user, respond with a JSON object with keys:
  - "type": the string "final_answer"
  - "answer": your natural language answer

Do NOT wrap the JSON in backticks.
Do NOT add any extra text before or after the JSON.
"""

LOOP_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT + "\n\n" + TOOL_DESCRIPTIONS),
        (
            "human",
            "Context:\n"
            "- user_id: {user_id}\n"
            "- intent: {intent}\n"
            "- slots: {slots}\n\n"
            "User message:\n"
            "{message}\n\n"
            "Previous steps:\n"
            "{history}\n"
            "Now decide what to do next: either call one tool, or give the final answer.\n",
        ),
    ]
)

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an HR assistant. Based on the user message and the tool results, "
            "produce a clear final answer. Do not mention tools or internal steps.",
        ),
        (
            "human",
            "User message:\n{message}\n\n"
            "Tool results:\n{history}\n",
        ),
    ]
)

llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    temperature=0.1,
    max_tokens=512,
)


def _safe_parse_json(content: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse JSON from the model output.
    Handles the case where the model might accidentally include backticks.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    try:
        return json.loads(text)
    except Exception:
        return None


def run_hr_react_agent(
    user_id: Optional[str],
    intent: str,
    slots: Dict[str, Any],
    message: str,
) -> str:
    """
    Minimal ReAct-style loop:
    - Up to 3 tool calls
    - Each step: Claude chooses either a tool_call or final_answer (JSON)
    - Tool outputs added to history
    - If no well-formed JSON, we treat whole output as final answer
    """
    uid = user_id or "user_123"

    history: List[str] = []

    for _ in range(3):
        msgs = LOOP_PROMPT.format_messages(
            user_id=uid,
            intent=intent,
            slots=slots,
            message=message,
            history="\n".join(history) if history else "(no previous steps)",
        )
        resp = llm.invoke(msgs)
        parsed = _safe_parse_json(resp.content)

        if not parsed:
            return resp.content

        if parsed.get("type") == "final_answer":
            return parsed.get("answer", "")

        if parsed.get("type") == "tool_call":
            tool_name = parsed.get("tool")
            args = parsed.get("arguments", {})

            tool = TOOLS.get(tool_name)
            if not tool:
                history.append(f"Tool error: unknown tool '{tool_name}'.")
                continue

            try:
                result = tool.invoke(args)
            except Exception as e:
                result = f"Tool '{tool_name}' failed with error: {e}"

            history.append(f"TOOL_CALL {tool_name}({args}) => {result}")
            continue

        # Unknown type: treat as final
        return resp.content

    # If we exit without final_answer, ask Claude to summarize
    msgs = FINAL_ANSWER_PROMPT.format_messages(
        message=message,
        history="\n".join(history) if history else "(no tool results)",
    )
    resp = llm.invoke(msgs)
    return resp.content
