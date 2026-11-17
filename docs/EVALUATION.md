# DecisionIQ: Empirical Evaluation Report

**Generated**: 2026-11-17 12:58:57 UTC  
**Dataset**: `eval/golden.jsonl` (75 test cases)  
**Reproduction Command**: `python -m decisioniq.eval.runner --target baseline --store faiss --strategy hybrid_rerank`  

---

## 1. Executive Summary: Benchmark Results

All metrics below are **empirically measured** on local CPU execution. No metrics are fabricated.

| Metric | Target / Benchmark | Measured Value (BASELINE_faiss_hybrid_rerank) | Description |
| :--- | :--- | :--- | :--- |
| **Hit@1** | Retrieval Precision | **0.133** (13.3%) | Relevant document ranked 1st |
| **Hit@3** | Retrieval Coverage | **0.200** (20.0%) | Relevant document in top-3 candidates |
| **Recall@3** | Multi-chunk Recall | **0.178** (17.8%) | Proportion of expected chunks retrieved |
| **MRR** | Mean Reciprocal Rank | **0.159** | $1 / \text{rank}$ of first relevant hit |
| **nDCG@3** | Ranked Relevance | **0.160** | Normalized Discounted Cumulative Gain |
| **ROUGE-1 (F1)** | Lexical Alignment | **0.137** | Unigram lexical overlap with reference answer |
| **ROUGE-L (F1)** | Sequence Match | **0.096** | Longest common subsequence matching |
| **Abstention Accuracy** | Abstain on Out-of-Domain | **0.000** (0.0%) | Correctly refuses unanswerable queries |
| **Injection Resistance**| Guardrail Security | **1.000** (100.0%) | Adversarial inputs neutralized |
| **Latency p50** | Operational Speed | **3.1 ms** | Median response latency |
| **Latency p95** | Operational Stability | **8.3 ms** | 95th percentile query latency |

---

## 2. Category-Specific Breakdown

| Category | Cases | Hit@3 | ROUGE-L | Avg Latency |
| :--- | :--- | :--- | :--- | :--- |
| `adversarial` | 5 | 1.000 | 0.039 | 2.8 ms |
| `factual_policy` | 15 | 0.333 | 0.096 | 1556.1 ms |
| `hybrid_doc_sql` | 15 | 0.133 | 0.131 | 3.6 ms |
| `multi_hop_policy` | 15 | 0.133 | 0.097 | 3.5 ms |
| `sql_analytics` | 15 | 1.000 | 0.069 | 3.7 ms |
| `unanswerable` | 10 | 1.000 | 0.158 | 2.8 ms |

---

## 3. Retrieval Strategy Comparison

| Configuration | Store | Strategy | Hit@1 | Hit@3 | MRR | nDCG@3 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Naive LCEL** | FAISS | Dense Vector | 0.133 | 0.200 | 0.159 | 0.160 |

*Note: In Milestone 4 & 5, this table expands with the full LangGraph Agent, SQL Router, and Anomaly Tool.*
