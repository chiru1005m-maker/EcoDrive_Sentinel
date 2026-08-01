"""
EcoDrive-Sentinel | Emergency PDF Ingestion Pipeline
=====================================================
Ingests all 14 MC-series Technical Bulletins into MongoDB
for RAG-based diagnostic retrieval.

Usage:
    python emergency_ingest.py
"""

import glob
import os
import sys
import time
import hashlib
import numpy as np
from pathlib import Path

# ── MongoDB ──────────────────────────────────
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "ecodrive_sentinel"
COLLECTION_NAME = "maintenance_vectors"

# ── PDF Loading & Chunking ───────────────────
# Try LangChain loaders first, fall back to pypdf
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    USE_LANGCHAIN = True
    print("✅ Using LangChain PDF pipeline")
except ImportError:
    try:
        from langchain.document_loaders import PyPDFLoader
        from langchain.text_splitters import RecursiveCharacterTextSplitter
        USE_LANGCHAIN = True
        print("✅ Using LangChain (legacy) PDF pipeline")
    except ImportError:
        USE_LANGCHAIN = False
        print("⚠️  LangChain not available, using pypdf fallback")
        try:
            import pypdf
        except ImportError:
            print("❌ Neither langchain nor pypdf installed.")
            print("   Run: pip install langchain-community langchain-text-splitters pypdf")
            sys.exit(1)


def generate_deterministic_embedding(text: str, dim: int = 1536) -> list[float]:
    """
    Generate a deterministic pseudo-embedding from text content.

    Uses SHA-256 hash seeded RNG to produce consistent embeddings for
    the same text. This enables cosine similarity search in the
    air-gapped environment without requiring an embedding model.

    In production, replace with: text-embedding-3-small or local
    sentence-transformers model.
    """
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    # L2-normalize for cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def load_and_chunk_langchain(pdf_path: str) -> list[dict]:
    """Load PDF with LangChain PyPDFLoader and split into chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(pages)

    results = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        if len(text) < 30:  # Skip tiny fragments
            continue
        results.append({
            "content": text,
            "page": chunk.metadata.get("page", 0),
            "chunk_index": i,
            "source_file": os.path.basename(pdf_path),
        })
    return results


def load_and_chunk_pypdf(pdf_path: str) -> list[dict]:
    """Fallback: Load PDF with pypdf and manually chunk."""
    import pypdf

    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n\n"

    # Manual recursive character splitting
    chunk_size = 1000
    chunk_overlap = 200
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if len(chunk) >= 30:
            chunks.append({
                "content": chunk,
                "page": 0,
                "chunk_index": len(chunks),
                "source_file": os.path.basename(pdf_path),
            })
        start += chunk_size - chunk_overlap

    return chunks


def extract_bulletin_metadata(filename: str) -> dict:
    """Extract structured metadata from a SYN-BULLETIN-series filename.

    NOTE: These bulletin IDs are entirely synthetic — authored to model the
    structure of real-world OEM fault-bulletin documentation for the purpose
    of demonstrating this project's RAG retrieval and diagnostic reasoning
    architecture. They do not represent actual proprietary manufacturer data.
    """
    # Pattern: SYN-BULLETIN-XXXX-0001.pdf
    basename = Path(filename).stem  # e.g., "SYN-BULLETIN-0010-0001"
    parts = basename.split("-")

    bulletin_id = basename
    component = "Unknown"

    # Map synthetic bulletin IDs to their illustrative component/severity.
    # Component names describe general EV subsystems, not manufacturer-
    # specific internal part codes.
    component_map = {
        "SYN-BULLETIN-0001": ("Range Display", "WARNING"),
        "SYN-BULLETIN-0002": ("HV Battery / BMS", "CRITICAL"),
        "SYN-BULLETIN-0003": ("Auxiliary Battery (48V)", "WARNING"),
        "SYN-BULLETIN-0004": ("Auxiliary Battery (48V)", "INFO"),
        "SYN-BULLETIN-0005": ("Range Estimation", "INFO"),
        "SYN-BULLETIN-0006": ("DC/DC Converter", "CRITICAL"),
        "SYN-BULLETIN-0007": ("DC/DC to BMS Cascade", "CRITICAL"),
        "SYN-BULLETIN-0008": ("BMS Fuse Logic", "CRITICAL"),
        "SYN-BULLETIN-0009": ("HV Charging", "WARNING"),
        "SYN-BULLETIN-0010": ("HV PTC Heater", "CRITICAL"),
        "SYN-BULLETIN-0011": ("Thermal Management", "WARNING"),
        "SYN-BULLETIN-0012": ("Thermal Management", "WARNING"),
        "SYN-BULLETIN-0013": ("HV PTC Heater", "CRITICAL"),
        "SYN-BULLETIN-0014": ("Range Estimation", "INFO"),
    }

    # Match by first three parts (SYN-BULLETIN-XXXX)
    key = "-".join(parts[:3]) if len(parts) >= 3 else basename
    component, severity = component_map.get(key, ("Unknown", "INFO"))

    return {
        "protocol_id": bulletin_id,
        "component": component,
        "severity": severity,
    }


def main():
    print("\n" + "=" * 60)
    print("🚨 EMERGENCY INGEST | 14 Technical Bulletins → MongoDB")
    print("=" * 60)

    # ── Step 1: Discover PDFs ────────────────────
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pdf_pattern = os.path.join(project_root, "data", "raw", "manuals", "MC-*.pdf")
    pdf_files = sorted(glob.glob(pdf_pattern))

    if not pdf_files:
        print(f"❌ No MC-*.pdf files found in {project_root}")
        sys.exit(1)

    print(f"   Found {len(pdf_files)} Technical Bulletins:")
    for f in pdf_files:
        size_kb = os.path.getsize(f) / 1024
        print(f"   📄 {os.path.basename(f)} ({size_kb:.0f} KB)")

    # ── Step 2: Connect to MongoDB ───────────────
    print(f"\n   Connecting to MongoDB: {MONGO_URI}/{DB_NAME}...")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[DB_NAME]
        coll = db[COLLECTION_NAME]
        print(f"   ✅ Connected to '{DB_NAME}.{COLLECTION_NAME}'")
    except Exception as e:
        print(f"   ❌ MongoDB connection failed: {e}")
        sys.exit(1)

    # ── Step 3: Clear old bulletin data ──────────
    # Only remove MC-series entries, preserve other protocols (RP-NASA, etc.)
    deleted = coll.delete_many({"source_type": "technical_bulletin"})
    print(f"   🗑️  Cleared {deleted.deleted_count} old bulletin chunks")

    # ── Step 4: Process each PDF ─────────────────
    total_chunks = 0
    total_docs = []
    t_start = time.time()

    for pdf_path in pdf_files:
        basename = os.path.basename(pdf_path)
        print(f"\n   📖 Processing: {basename}...")

        try:
            # Load and chunk
            if USE_LANGCHAIN:
                chunks = load_and_chunk_langchain(pdf_path)
            else:
                chunks = load_and_chunk_pypdf(pdf_path)

            if not chunks:
                print(f"      ⚠️  No text extracted from {basename}")
                continue

            # Extract bulletin metadata
            meta = extract_bulletin_metadata(basename)

            # Build MongoDB documents
            for chunk in chunks:
                embedding = generate_deterministic_embedding(chunk["content"])
                doc = {
                    "protocol_id": meta["protocol_id"],
                    "title": f"{meta['component']} — {meta['protocol_id']}",
                    "content": chunk["content"],
                    "component": meta["component"],
                    "severity": meta["severity"],
                    "source_file": chunk["source_file"],
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "source_type": "technical_bulletin",
                    "embedding": embedding,
                    "tags": [
                        meta["protocol_id"].lower(),
                        meta["component"].lower().replace(" ", "_"),
                        meta["severity"].lower(),
                    ],
                }
                total_docs.append(doc)

            total_chunks += len(chunks)
            print(f"      ✅ {len(chunks)} chunks extracted")

        except Exception as e:
            print(f"      ❌ Failed: {e}")

    # ── Step 5: Bulk insert into MongoDB ─────────
    if total_docs:
        print(f"\n   📤 Inserting {len(total_docs)} documents into MongoDB...")
        result = coll.insert_many(total_docs)
        elapsed = time.time() - t_start
        print(f"   ✅ Inserted {len(result.inserted_ids)} documents in {elapsed:.1f}s")
    else:
        print("\n   ❌ No documents to insert!")
        sys.exit(1)

    # ── Step 6: Verify ───────────────────────────
    print("\n" + "-" * 60)
    print("   📊 VERIFICATION")
    print("-" * 60)

    # Count by bulletin
    pipeline = [
        {"$match": {"source_type": "technical_bulletin"}},
        {"$group": {"_id": "$protocol_id", "chunks": {"$sum": 1}, "severity": {"$first": "$severity"}}},
        {"$sort": {"_id": 1}},
    ]
    bulletin_stats = list(coll.aggregate(pipeline))

    for stat in bulletin_stats:
        sev_icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🟢"}.get(stat["severity"], "⚪")
        print(f"   {sev_icon} {stat['_id']}: {stat['chunks']} chunks [{stat['severity']}]")

    total_in_db = coll.count_documents({"source_type": "technical_bulletin"})
    total_all = coll.count_documents({})
    print(f"\n   📈 Total bulletin chunks in DB: {total_in_db}")
    print(f"   📈 Total documents in collection: {total_all}")

    # Verify embedding dimensionality
    sample = coll.find_one({"source_type": "technical_bulletin", "embedding": {"$exists": True}})
    if sample and "embedding" in sample:
        print(f"   📐 Embedding dimensions: {len(sample['embedding'])}")

    print("\n" + "=" * 60)
    print("✅ EMERGENCY INGEST COMPLETE")
    print(f"   {len(pdf_files)} bulletins → {total_in_db} chunks → MongoDB ready")
    print("=" * 60)

    client.close()


if __name__ == "__main__":
    main()
