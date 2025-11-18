import os
from typing import List

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
POLICIES_DIR = os.path.join(BASE_DIR, "data", "hr_policies")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_hr")


def build_or_load_vectorstore():
    os.makedirs(POLICIES_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    # Seed demo file if empty
    if not os.listdir(POLICIES_DIR):
        with open(
            os.path.join(POLICIES_DIR, "demo_policy.md"), "w", encoding="utf-8"
        ) as f:
            f.write(
                "Demo HR policy: Employees have 15 days of PTO per year. "
                "Parental leave is 16 weeks paid. "
                "Sick leave is 10 days per year.\n"
            )

    texts: List[str] = []
    metadatas: List[dict] = []

    for fname in os.listdir(POLICIES_DIR):
        path = os.path.join(POLICIES_DIR, fname)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        texts.append(text)
        metadatas.append({"source": fname})

    # Local embedding model, no external API
    embeddings = FastEmbedEmbeddings()

    # For simplicity, rebuild on startup; Chroma will persist to disk
    vs = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=CHROMA_DIR,
    )
    vs.persist()
    return vs
