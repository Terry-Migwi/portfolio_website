from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY")
PINECONE_API_KEY   = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST      = os.environ.get("PINECONE_HOST")

NAMESPACE = "portfolio"


def embed_and_query(question: str) -> list:
    # Use Pinecone's inference API to embed the question
    embed_url = "https://api.pinecone.io/embed"
    embed_payload = json.dumps({
        "model": "multilingual-e5-large",
        "inputs": [{"text": question}],
        "parameters": {
            "input_type": "query",
            "truncate": "END"
        }
    }).encode("utf-8")

    embed_req = urllib.request.Request(
        embed_url,
        data=embed_payload,
        headers={
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2024-10"
        }
    )
    with urllib.request.urlopen(embed_req) as response:
        embed_data = json.loads(response.read())

    vector = embed_data["data"][0]["values"]

    # Query Pinecone with the embedding
    query_url = f"https://{PINECONE_HOST}/query"
    query_payload = json.dumps({
        "vector": vector,
        "topK": 5,
        "namespace": NAMESPACE,
        "includeMetadata": True
    }).encode("utf-8")

    query_req = urllib.request.Request(
        query_url,
        data=query_payload,
        headers={
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(query_req) as response:
        query_data = json.loads(response.read())

    return query_data.get("matches", [])


def build_context(matches: list) -> str:
    chunks = []
    for i, match in enumerate(matches):
        filename = match.get("metadata", {}).get("filename", "unknown")
        text     = match.get("metadata", {}).get("text", "")
        chunks.append(f"[Source {i + 1}: {filename}]\n{text}")
    return "\n\n---\n\n".join(chunks)


def call_claude(messages: list, context: str) -> str:
    system_prompt = f"""You are an assistant on Terry Migwi's portfolio website. Your job is to answer questions about Terry's projects only, using the retrieved context below.

If someone asks about anything outside of these projects, including Terry's personal background, availability, salary, or anything else not covered in the context, respond with: 
"I am sorry, my instructions are to stay within the scope of these projects. If you would like more information on this, feel free to contact Terry."
Important: Denta is a messaging assistant that works over both WhatsApp and SMS channels. Do not describe it as WhatsApp-only. Always refer to it as a messaging assistant.

Do not make up any information not present in the context. Keep responses concise and professional.

Portfolio overview:
This portfolio contains four projects:
1. Denta - a messaging assistant with RAG and tool calling for bookings, designed for any business taking bookings or appointments over a messaging channel.
2. Hybrid Search - a production-ready document search engine combining semantic search and keyword search with LLM answer synthesis, secured with JWT authentication.
3. TournamentIQ - a live AI match intelligence platform built on 2026 FIFA World Cup data, combining Elo ratings, Poisson probability modelling, Bayesian inference, and LangGraph orchestration to generate narrative match intelligence briefs.
4. Attendance Tracker - a Python automation that reduced weekly learner attendance tracking from 2 hours to under 10 minutes.

Retrieved context:
{context}"""

    url = "https://api.anthropic.com/v1/messages"
    payload = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": messages
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


class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload  = json.loads(body)
            messages = payload.get("messages", [])

            if not messages:
                self._respond(400, {"error": "Invalid request body"})
                return

            matches = embed_and_query(messages[-1]["content"])
            context = build_context(matches)
            reply   = call_claude(messages, context)

            self._respond(200, {"reply": reply})

        except Exception as e:
            self._respond(500, {"error": str(e)})

    def do_OPTIONS(self):
        self._respond(200, {})

    def _respond(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
