from pathlib import Path
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import json
import urllib.request

load_dotenv()

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY")
PINECONE_API_KEY   = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST      = os.environ.get("PINECONE_HOST")
NAMESPACE          = "portfolio"


def embed_and_query(question: str) -> list:
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Step 1 — embed the question using Pinecone inference
    print("   Embedding question with Pinecone inference...")
    embeddings = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[question],
        parameters={"input_type": "query", "truncate": "END"}
    )
    vector = embeddings[0].values
    print(f"   Vector length: {len(vector)}")

    # Step 2 — query Pinecone
    print("   Querying Pinecone...")
    index   = pc.Index(host=PINECONE_HOST)
    results = index.query(
        vector=vector,
        top_k=5,
        namespace=NAMESPACE,
        include_metadata=True
    )
    matches = results.get("matches", [])
    print(f"   Retrieved {len(matches)} chunks")
    return matches


def build_context(matches: list) -> str:
    chunks = []
    for i, match in enumerate(matches):
        filename = match.get("metadata", {}).get("filename", "unknown")
        text     = match.get("metadata", {}).get("text", "")
        chunks.append(f"[Source {i + 1}: {filename}]\n{text}")
    return "\n\n---\n\n".join(chunks)


def call_claude(question: str, context: str) -> str:
    system_prompt = f"""You are an assistant on Terry Migwi's portfolio website. Answer questions about Terry's projects only using the context below.

If the question is outside the projects, say: "I am sorry, I do not have an answer for that. Feel free to contact Terry."

Important: Denta is a messaging assistant that works over both WhatsApp and SMS channels. Do not describe it as WhatsApp-only. Always refer to it as a messaging assistant.

Portfolio overview:
This portfolio contains four projects:
1. Denta - a messaging assistant with RAG and tool calling for bookings, designed for any business taking bookings or appointments over a messaging channel.
2. Hybrid Search - a production-ready document search engine combining semantic search and keyword search with LLM answer synthesis, secured with JWT authentication.
3. TournamentIQ - a live AI match intelligence platform built on 2026 FIFA World Cup data, combining Elo ratings, Poisson probability modelling, Bayesian inference, and LangGraph orchestration to generate narrative match intelligence briefs.
4. Attendance Tracker - a Python automation that reduced weekly learner attendance tracking from 2 hours to under 10 minutes.

Retrieved context:
{context}"""

    url     = "https://api.anthropic.com/v1/messages"
    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": [{"role": "user", "content": question}]
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read())

    return data["content"][0]["text"]


def chat(question: str):
    print(f"\nQuestion: {question}")
    print("-" * 50)

    matches = embed_and_query(question)
    context = build_context(matches)
    print(f"   Context length: {len(context)} characters")

    reply = call_claude(question, context)
    print(f"\nAnswer: {reply}")
    print("=" * 50)


if __name__ == "__main__":
    chat("How many projects are in this portfolio?")
    chat("Tell me about the messaging assistant project")
    chat("How does the Bayesian layer work in TournamentIQ?")