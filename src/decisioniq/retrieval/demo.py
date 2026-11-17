"""Interactive CLI demonstration for DecisionIQ retrieval strategies and baseline RAG."""

import argparse

from decisioniq.config import settings
from decisioniq.retrieval import get_retrieval_pipeline
from decisioniq.retrieval.baseline_rag import BaselineRAGChain


def run_demo(
    query: str,
    store: str = "faiss",
    strategy: str = "hybrid_rerank",
    top_k: int = 3,
    run_rag: bool = True,
):
    print("=" * 70)
    print("DecisionIQ Retrieval & RAG Demonstration")
    print("=" * 70)
    print(f"Query:    '{query}'")
    print(f"Store:    {store.upper()}")
    print(f"Strategy: {strategy}")
    print(f"Top-K:    {top_k}")
    print("-" * 70)

    pipeline = get_retrieval_pipeline(
        store_type=store,  # type: ignore
        strategy=strategy,  # type: ignore
    )

    results = pipeline.retrieve(query, top_k_override=top_k)
    print(f"\nRetrieved {len(results)} Chunks:")
    for rank, (doc, score) in enumerate(results, 1):
        cid = doc.metadata.get("chunk_id", "unknown")
        title = doc.metadata.get("title", doc.metadata.get("filename", "Untitled"))
        status = doc.metadata.get("status", "unknown")
        snippet = doc.page_content.replace("\n", " ")[:140]
        print(f"  [{rank}] Score: {score:.4f} | ID: {cid} ({status}) | Title: {title}")
        print(f"      Snippet: {snippet}...")

    if run_rag:
        print("\n" + "-" * 70)
        print("Synthesizing Answer via Baseline RAG Chain:")
        rag_chain = BaselineRAGChain(retrieval_pipeline=pipeline)
        resp = rag_chain.invoke(query)
        print("\nAnswer:")
        print(resp["answer"])

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="DecisionIQ Retrieval Demo CLI")
    parser.add_argument(
        "--query",
        type=str,
        default="What is the overtime rule for employees in grade L3?",
        help="Business query to retrieve against",
    )
    parser.add_argument(
        "--store",
        choices=["faiss", "chroma"],
        default=settings.vector_store,
        help="Vector store to query",
    )
    parser.add_argument(
        "--strategy",
        choices=["dense", "bm25", "hybrid", "hybrid_rerank"],
        default="hybrid_rerank",
        help="Retrieval strategy",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip LLM answer generation and only show retrieved chunks",
    )

    args = parser.parse_args()
    run_demo(
        query=args.query,
        store=args.store,
        strategy=args.strategy,
        top_k=args.top_k,
        run_rag=not args.no_rag,
    )


if __name__ == "__main__":
    main()
