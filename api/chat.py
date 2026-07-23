from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.error
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY")
PINECONE_API_KEY   = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST      = os.environ.get("PINECONE_HOST")
HUGGINGFACE_API_KEY = os.environ.get("HUGGING_FACE_HUB_KEY")

NAMESPACE = "portfolio"


def embed_question(question: str) -> list:
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    payload = json.dumps({
        "inputs": question,
        "options": {"wait_for_model": True}
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(req) as response:
        embedding = json.loads(response.read())

    return embedding[0] if isinstance(embedding[0], list) else embedding


def query_pinecone(vector: list) -> list:
    url = f"https://{PINECONE_HOST}/query"
    payload = json.dumps({
        "vector": vector,
        "topK": 5,
        "namespace": NAMESPACE,
        "includeMetadata": True
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Api-Key": PINECONE_API_KEY,
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read())

    return data.get("matches", [])


def build_context(matches: list) -> str:
    chunks = []
    for i, match in enumerate(matches):
        filename = match.get("metadata", {}).get("filename", "unknown")
        text     = match.get("metadata", {}).get("text", "")
        chunks.append(f"[Source {i + 1}: {filename}]\n{text}")
    return "\n\n---\n\n".join(chunks)


def call_claude(messages: list, context: str) -> str:
    system_prompt = f"""You are an assistant on Terry Migwi's portfolio website. Your job is to answer questions about Terry's projects using the retrieved context below.

If someone asks about anything outside of these projects, including Terry's personal background, availability, salary, or anything else not covered in the context, respond with: "I am sorry, I do not have an answer for that. Feel free to contact Terry."

Do not make up any information not present in the context. Keep responses concise and professional.

Portfolio overview:
This portfolio contains four projects:
1. Denta — a production-grade messaging assistant with RAG and tool calling for bookings, built for a dental clinic but designed for any business taking bookings over a messaging channel.
2. Hybrid Search — a production-ready document search engine combining semantic search and keyword search with LLM answer synthesis, secured with JWT authentication.
3. TournamentIQ — a live AI match intelligence platform built on 2026 FIFA World Cup data, combining Elo ratings, Poisson probability modelling, Bayesian inference, and LangGraph orchestration to generate narrative match intelligence briefs.
4. Attendance Tracker — a Python automation that reduced weekly learner attendance tracking from 2 hours to under 10 minutes.

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
            payload = json.loads(body)
            messages = payload.get("messages", [])

            if not messages:
                self._respond(400, {"error": "Invalid request body"})
                return

            user_question = messages[-1]["content"]

            vector  = embed_question(user_question)
            matches = query_pinecone(vector)
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