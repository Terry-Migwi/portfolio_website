from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import uuid
import os
from dotenv import load_dotenv
load_dotenv()

DATA_DIR   = Path("data")
INDEX_NAME = "dental-clinic"
NAMESPACE  = "portfolio"
DIMENSION  = 384

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def get_pinecone_client():
    return Pinecone(api_key=PINECONE_API_KEY)


def get_ingested_filenames(pc: Pinecone) -> set:
    try:
        index = pc.Index(INDEX_NAME)
        results = index.query(
            vector=[0.0] * DIMENSION,
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


def load_documents(data_dir: Path) -> list[Document]:
    docs = []
    for filepath in sorted(data_dir.iterdir()):
        if filepath.suffix != ".txt":
            continue
        print(f"   Loading: {filepath.name}")
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(
            page_content=text,
            metadata={
                "filename": filepath.name,
                "source": str(filepath)
            }
        ))
    return docs


def upsert_chunks(
    pc: Pinecone,
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    batch_size: int = 50
):
    index  = pc.Index(INDEX_NAME)
    texts  = [chunk.page_content for chunk in chunks]

    print("4. Generating dense embeddings...")
    dense_vectors = embeddings.embed_documents(texts)
    print(f"   → {len(dense_vectors)} dense vectors generated\n")

    print("5. Upserting to Pinecone...")
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": dense_vectors[i],
            "metadata": {
                "text": chunk.page_content,
                "filename": chunk.metadata.get("filename", "unknown"),
                "project": chunk.metadata.get("filename", "").split("_")[0],
            }
        })

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i: i + batch_size]
        index.upsert(vectors=batch, namespace=NAMESPACE)
        print(f"   → Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

    print(f"\n   → {len(vectors)} chunks upserted total\n")


def ingest():
    print("=== Portfolio Ingestion Pipeline ===\n")

    pc = get_pinecone_client()

    print("1. Checking already-ingested files...")
    already_ingested = get_ingested_filenames(pc)
    print(f"   → Already ingested: {already_ingested or 'none'}\n")

    print("2. Loading documents from disk...")
    all_docs = load_documents(DATA_DIR)

    new_docs = [
        doc for doc in all_docs
        if doc.metadata.get("filename") not in already_ingested
    ]

    if not new_docs:
        print("   → No new files to ingest. All up to date.")
        return

    print(f"   → {len(new_docs)} new document(s) to ingest\n")

    print("3. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    new_chunks = splitter.split_documents(new_docs)
    print(f"   → {len(new_chunks)} chunks created\n")

    embeddings = get_embeddings()
    upsert_chunks(pc, new_chunks, embeddings)

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()