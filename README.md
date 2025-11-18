# HR Workflow Automation Agent (Docker + LangChain + LangGraph + Anthropic Claude)

A fully functional multi-step HR automation agent built using:

-   **LangChain** (tools, prompts, RAG)
-   **LangGraph** (state machine flow)
-   **Anthropic Claude** (LLM)
-   **FastAPI** (HTTP API)
-   **Docker** (containerized app)
-   **Heuristic + LLM hybrid routing** (intent classifier + REACT loop)

This agent replicates Moveworks-style enterprise HR automations with
deterministic routing + LLM reasoning.

## 🚀 Features

### **1. PTO Balance Lookup**

Deterministic route.

### **2. HR Policy Lookup (RAG)**

### **3. Profile Updates**

### **4. Hardware Requests**

### **5. Anthropic Claude REACT Agent**

## 🧠 Architecture

    [Flow diagram…]

## 🐳 Running with Docker

``` bash
docker compose build
docker compose up
```

## 📁 Project Structure

    app/
      ├── main.py
      ├── graph.py
      ├── agent_react.py
      ├── tools.py
      ├── rag_index.py
      ├── config.py
      └── policies/

## 🔑 Environment Variables

    ANTHROPIC_API_KEY=your-key
    LANGCHAIN_API_KEY=your-key
    LANGCHAIN_TRACING_V2="false"

## ⭐ Example Responses

``` json
{ "route": "pto_balance", "answer": "You have 18.5 days of PTO remaining." }
```
