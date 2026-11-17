"""Synthetic triplet generator for embedding model fine-tuning.

Generates (query, positive_chunk, hard_negative_chunk) tuples from domain documents.
Hard negatives are sampled from related documents in the same domain or outdated policy versions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from decisioniq.ingestion.chunker import DocumentChunker
from decisioniq.ingestion.loaders import load_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Domain query templates per document category
DOMAIN_QUERY_PAIRS = [
    (
        "leave",
        [
            ("How many privilege leaves can be carried forward to next year?", "carry forward"),
            ("What is the maternity leave entitlement and duration?", "maternity leave"),
            ("Can paternity leave be taken in multiple installments?", "paternity leave"),
            ("How many days of bereavement leave are granted for immediate family?", "bereavement"),
            ("Are employees on probation eligible for casual leave?", "probation leave"),
            ("What happens to unutilized leave balances upon resignation?", "encashment"),
        ],
    ),
    (
        "overtime",
        [
            ("What is the overtime multiplier for L1 associates working weekends?", "grade l1"),
            ("Does a grade L3 specialist need prior approval for overtime?", "grade l3"),
            ("Are managers in grade L5 eligible for overtime pay?", "grade l5"),
            ("How are holiday hours compensated under the overtime policy?", "holiday"),
            ("What is the maximum allowed overtime hours per week?", "maximum overtime"),
            ("How is compensatory off granted in lieu of overtime pay?", "compensatory off"),
        ],
    ),
    (
        "expense",
        [
            ("What is the per diem travel allowance limit for London?", "per diem london"),
            ("What receipt documentation is mandatory for meal expenses?", "receipt"),
            ("What is the maximum flight class allowed for domestic travel?", "economy class"),
            ("Within how many days must travel expense claims be submitted?", "30 days"),
            ("Can personal entertainment be claimed under business expenses?", "non-reimbursable"),
        ],
    ),
    (
        "it_security",
        [
            ("What is the mandatory password rotation period?", "password rotation"),
            ("Are employees permitted to plug personal USB drives into work laptops?", "usb storage"),
            ("What is the protocol for reporting a suspected data phishing incident?", "phishing"),
            ("Is multi-factor authentication mandatory for remote VPN access?", "mfa"),
            ("How should confidential source code repositories be handled?", "source code"),
        ],
    ),
    (
        "timesheet",
        [
            ("What is the weekly deadline for timesheet submission?", "submission deadline"),
            ("What are the consequences of late timesheet approval by managers?", "manager approval"),
            ("How should project codes be assigned for internal bench time?", "bench project"),
            ("Can an employee log more than 12 hours in a single calendar day?", "daily limit"),
            ("Who audits flagged duplicate timesheet entries?", "audit"),
        ],
    ),
    (
        "payroll",
        [
            ("What is the standard monthly payroll cutoff date?", "cutoff date"),
            ("How is pro-rated salary computed for mid-month joiners?", "pro-rated"),
            ("What deductions are made for unpaid leaves of absence?", "unpaid leave"),
            ("When are annual performance bonuses credited to payroll?", "performance bonus"),
            ("How are tax withholdings adjusted for overseas deputation?", "tax withholding"),
        ],
    ),
]


def generate_triplets(
    docs_dir: Path = Path("data/docs"),
    output_dir: Path = Path("finetune/data"),
    train_ratio: float = 0.8,
) -> Tuple[int, int]:
    """Generates synthetic triplet datasets with hard negatives."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and chunk documents
    raw_docs = load_directory(docs_dir)
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=40)
    chunks = chunker.chunk_documents(raw_docs)
    logger.info("Loaded %d chunks from %d docs for triplet generation", len(chunks), len(raw_docs))

    # Categorize chunks by filename keyword
    chunks_by_category: Dict[str, List[Tuple[str, str, Dict]]] = {
        "leave": [],
        "overtime": [],
        "expense": [],
        "it_security": [],
        "timesheet": [],
        "payroll": [],
        "general": [],
    }

    for c in chunks:
        fn = c.metadata.get("filename", "").lower()
        content = c.page_content
        chunk_id = c.metadata.get("chunk_id", "chunk")
        meta = c.metadata

        placed = False
        for cat in ["leave", "overtime", "expense", "it_security", "timesheet", "payroll"]:
            if cat in fn:
                chunks_by_category[cat].append((chunk_id, content, meta))
                placed = True
                break
        if not placed:
            chunks_by_category["general"].append((chunk_id, content, meta))

    triplets = []

    # 2. Match queries to positive and hard-negative chunks
    for cat, query_specs in DOMAIN_QUERY_PAIRS:
        cat_chunks = chunks_by_category.get(cat, [])
        if not cat_chunks:
            continue

        for query, keyword in query_specs:
            # Find positive chunk matching keyword
            pos_chunk = None
            for cid, content, meta in cat_chunks:
                if keyword in content.lower():
                    pos_chunk = (cid, content, meta)
                    break
            if not pos_chunk and cat_chunks:
                pos_chunk = cat_chunks[0]

            if not pos_chunk:
                continue

            # Pick hard negative: another chunk from SAME category or version, but NOT the positive chunk
            hard_neg_chunk = None
            for cid, content, meta in cat_chunks:
                if cid != pos_chunk[0] and keyword not in content.lower():
                    hard_neg_chunk = (cid, content, meta)
                    break

            # If none in same category, pick from related category (e.g. payroll vs expense)
            if not hard_neg_chunk:
                for other_cat in chunks_by_category:
                    if other_cat != cat and chunks_by_category[other_cat]:
                        hard_neg_chunk = chunks_by_category[other_cat][0]
                        break

            if pos_chunk and hard_neg_chunk:
                triplets.append(
                    {
                        "query": query,
                        "positive": pos_chunk[1],
                        "positive_chunk_id": pos_chunk[0],
                        "positive_metadata": pos_chunk[2],
                        "negative": hard_neg_chunk[1],
                        "negative_chunk_id": hard_neg_chunk[0],
                        "category": cat,
                    }
                )

    # 3. Train/Val split
    split_idx = int(len(triplets) * train_ratio)
    train_triplets = triplets[:split_idx]
    val_triplets = triplets[split_idx:]

    train_path = output_dir / "triplets_train.jsonl"
    val_path = output_dir / "triplets_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for t in train_triplets:
            f.write(json.dumps(t) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for t in val_triplets:
            f.write(json.dumps(t) + "\n")

    logger.info("Generated %d training triplets -> %s", len(train_triplets), train_path)
    logger.info("Generated %d validation triplets -> %s", len(val_triplets), val_path)
    return len(train_triplets), len(val_triplets)


if __name__ == "__main__":
    generate_triplets()
