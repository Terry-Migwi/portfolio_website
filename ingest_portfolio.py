from pathlib import Path
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import uuid

load_dotenv()

DATA_DIR        = Path("data")
INDEX_NAME      = "portfolio"
NAMESPACE       = "portfolio"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


def get_pinecone_client():
    return Pinecone(api_key=PINECONE_API_KEY)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_documents(data_dir: Path) -> list:
    docs = []
    for filepath in sorted(data_dir.iterdir()):
        if filepath.suffix != ".txt":
            continue
        print(f"   Loading: {filepath.name}")
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        docs.append({
            "filename": filepath.name,
            "text": text
        })
    return docs


def get_ingested_filenames(index) -> set:
    try:
        results = index.query(
            vector=[0.0] * 768,
            top_k=10000,
            namespace=NAMESPACE,
            include_metadata=True
        )
        return {
            match.get("metadata", {}).get("filename")
            for match in results.get("matches", [])
            if match.get("metadata", {}).get("filename")
        }
    except Exception:
        return set()


def ingest():
    print("=== Portfolio Ingestion Pipeline (Pinecone Inference) ===\n")

    pc    = get_pinecone_client()
    index = pc.Index(INDEX_NAME)

    print("1. Checking already ingested files...")
    already_ingested = get_ingested_filenames(index)
    print(f"   Already ingested: {already_ingested or 'none'}\n")

    print("2. Loading documents...")
    all_docs = load_documents(DATA_DIR)

    new_docs = [
        doc for doc in all_docs
        if doc["filename"] not in already_ingested
    ]

    if not new_docs:
        print("   No new files to ingest. All up to date.")
        return

    print(f"   {len(new_docs)} new document(s) to ingest\n")

    print("3. Chunking documents...")
    all_chunks = []
    for doc in new_docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append({
                "id":       str(uuid.uuid4()),
                "text":     chunk,
                "filename": doc["filename"]
            })
    print(f"   {len(all_chunks)} chunks created\n")

    print("4. Embedding and upserting to Pinecone...")
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch  = all_chunks[i: i + batch_size]
        texts  = [chunk["text"] for chunk in batch]

        embeddings = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"}
        )

        vectors = []
        for j, chunk in enumerate(batch):
            vectors.append({
                "id":     chunk["id"],
                "values": embeddings[j].values,
                "metadata": {
                    "text":     chunk["text"],
                    "filename": chunk["filename"]
                }
            })

        index.upsert(vectors=vectors, namespace=NAMESPACE)
        print(f"   Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

    print(f"\nIngestion complete. {len(all_chunks)} chunks upserted.")


if __name__ == "__main__":
    ingest()