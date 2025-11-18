# app/graph.py

from typing import TypedDict, Literal, Dict, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver

from .config import ANTHROPIC_MODEL
from .agent_react import run_hr_react_agent


# ===========
# State
# ===========

class HRState(TypedDict, total=False):
    user_id: str
    user_input: str
    intent: Literal[
        "pto_balance",
        "policy_lookup",
        "profile_update",
        "hardware_request",
        "fallback_ticket",
    ]
    slots: Dict[str, Any]
    answer: str


llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    temperature=0.1,
    max_tokens=256,
)


# ===========
# Intent classification node
# ===========

intent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an HR assistant. Classify user intent strictly into one of:\n"
            "[pto_balance, policy_lookup, profile_update, hardware_request, fallback_ticket].\n\n"
            "Use these guidelines:\n"
            "- pto_balance: questions about time off, PTO, vacation days, sick days.\n"
            "- policy_lookup: questions about HR policies, benefits, parental leave, handbooks.\n"
            "- profile_update: requests to update personal or emergency contact information.\n"
            "- hardware_request: requests for laptops, computers, monitors, keyboards, or other equipment.\n"
            "- fallback_ticket: anything else.\n"
            "Return ONLY the label, nothing else.",
        ),
        ("user", "{text}"),
    ]
)

def classify_intent_node(state: HRState) -> HRState:
    text = state["user_input"]
    lower = text.lower()

    intent: HRState["intent"]

    # Heuristic first (cheap + deterministic)
    if any(k in lower for k in ["pto", "vacation", "time off", "leave balance", "sick leave"]):
        intent = "pto_balance"
    elif any(k in lower for k in ["policy", "handbook", "parental leave", "benefits"]):
        intent = "policy_lookup"
    elif any(
        k in lower
        for k in ["emergency contact", "phone number", "update my info", "update my details"]
    ):
        intent = "profile_update"
    elif any(
        k in lower
        for k in [
            "laptop",
            "macbook",
            "computer",
            "pc",
            "monitor",
            "keyboard",
            "mouse",
            "docking station",
            "new equipment",
            "new hardware",
            "hardware request",
        ]
    ):
        intent = "hardware_request"
    else:
        # Fallback to LLM classifier
        msg = intent_prompt.format_messages(text=text)
        resp = llm.invoke(msg)
        intent = resp.content.strip()  # type: ignore

        # Guardrail: snap unknown labels to fallback_ticket
        allowed = {
            "pto_balance",
            "policy_lookup",
            "profile_update",
            "hardware_request",
            "fallback_ticket",
        }
        if intent not in allowed:
            intent = "fallback_ticket"

    state["intent"] = intent
    if "slots" not in state:
        state["slots"] = {}
    return state

# ===========
# Slot filling node (for profile updates)
# ===========

slot_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are helping collect structured info for an HR profile update.\n"
            "Given the text, extract fields as JSON with keys: name, phone, relationship.\n"
            "If a field is missing, set it to null. Return ONLY the JSON.",
        ),
        ("user", "{text}"),
    ]
)


def collect_slots_node(state: HRState) -> HRState:
    # Only relevant for profile_update intents
    if state.get("intent") != "profile_update":
        return state

    text = state["user_input"]
    msgs = slot_prompt.format_messages(text=text)
    resp = llm.invoke(msgs)

    import json

    try:
        parsed = json.loads(resp.content)
    except Exception:
        parsed = {"name": None, "phone": None, "relationship": None}

    slots = state.get("slots", {})
    slots.update(parsed)
    state["slots"] = slots
    return state


# ===========
# ReAct agent node
# ===========

def hr_react_node(state: HRState) -> HRState:
    """
    Hybrid step: use graph for intent/slots, then have a ReAct agent
    actually decide which tools to call and produce the final answer.
    """
    user_id = state.get("user_id")
    intent = state.get("intent", "fallback_ticket")
    slots = state.get("slots", {})
    message = state["user_input"]

    answer = run_hr_react_agent(
        user_id=user_id,
        intent=intent,
        slots=slots,
        message=message,
    )
    state["answer"] = answer
    return state


# ===========
# Build LangGraph
# ===========

graph_builder = StateGraph(HRState)

graph_builder.add_node("classify_intent", classify_intent_node)
graph_builder.add_node("collect_slots", collect_slots_node)
graph_builder.add_node("hr_react", hr_react_node)

graph_builder.set_entry_point("classify_intent")


def route_after_intent(state: HRState) -> str:
    """
    If we need slots (profile_update), go to slot filler; otherwise, go straight
    to ReAct agent node.
    """
    intent = state["intent"]
    if intent == "profile_update":
        return "collect_slots"
    return "hr_react"


graph_builder.add_conditional_edges(
    "classify_intent",
    route_after_intent,
    {
        "collect_slots": "collect_slots",
        "hr_react": "hr_react",
    },
)

# After slot collection, always go to ReAct agent
graph_builder.add_edge("collect_slots", "hr_react")

# ReAct node is terminal
graph_builder.add_edge("hr_react", END)

# memory = MemorySaver()
hr_app = graph_builder.compile()
