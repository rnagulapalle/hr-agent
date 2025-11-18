from typing import Dict, Any
from langchain_core.tools import tool
from .rag_index import build_or_load_vectorstore

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_or_load_vectorstore()
    return _vectorstore


# --- HRIS / PTO tools (mock implementations) ---

@tool
def get_pto_balance(user_id: str) -> str:
    """Get the PTO balance for a user_id (mock HRIS)."""
    # In real life, call Workday / BambooHR / etc.
    mock_balances = {
        "user_123": "12 days",
        "user_raj": "18.5 days",
    }
    return mock_balances.get(user_id, "10 days")  # default for demo


@tool
def update_emergency_contact(user_id: str, name: str, phone: str, relationship: str) -> str:
    """Update emergency contact for a user (mock)."""
    # Real impl: POST to HRIS API.
    return (
        f"Updated emergency contact for {user_id} to {name} ({relationship}), "
        f"phone {phone}."
    )


@tool
def create_hr_ticket(user_id: str, summary: str, details: str) -> str:
    """Create an HR ticket (mock ServiceNow/Jira)."""
    # Real impl: call ServiceNow/Jira.
    return f"Created HR ticket TCKT-1234 for {user_id}: {summary}"


@tool
def search_hr_policies(query: str) -> str:
    """Search HR policy documents and return top chunks."""
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=3)
    if not docs:
        return "No relevant policy found."
    joined = "\n\n---\n\n".join(
        f"[{d.metadata.get('source')}] {d.page_content}" for d in docs
    )
    return joined

@tool
def create_hardware_request(user_id: str, item: str, justification: str) -> str:
    """Create a hardware/equipment request for the user (mock)."""
    # Real impl would call your ITSM/hardware system.
    # Here we just return a fake ID.
    return (
        f"Created hardware request HW-1234 for {user_id}: {item} "
        f"(reason: {justification})."
    )
