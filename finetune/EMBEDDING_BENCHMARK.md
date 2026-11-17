# DecisionIQ: Embedding Fine-Tuning Benchmark Report

**Generated**: 2026-11-17 21:29:21 UTC
**Base Architecture**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
**Loss Objective**: `MultipleNegativesRankingLoss` with In-Batch Hard Negatives
**Optimization**: Triplet Margin Adaptation (Orthogonal Procrustes Projection), 3 Epochs
**Training Hardware**: CPU Laptop Execution (Wall Clock: 0.4s)

---

## 1. Measured Retrieval Results (Held-Out Validation Set)

The validation split consists of **6 domain query-positive-negative triplets** evaluated against the full **16-passage document corpus**.

| Metric | Base Embedding Model | Domain Fine-Tuned Model | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Hit@1** | `0.0%` | `0.0%` | `+0.0%` | `+0.0%` |
| **Hit@3** | `16.7%` | `16.7%` | `+0.0%` | `+0.0%` |
| **MRR (Mean Reciprocal Rank)** | `0.1771` | `0.1771` | `+0.0000` | `+0.0%` |

---

## 2. Key Observations & Honest Engineering Trade-offs

1. **Why Domain Adaptation Matters**:
   - Out-of-the-box embeddings often confuse domain terminology across closely related policies (e.g. *Privilege Leave carry-forward* vs *Casual Leave lapse rules*).
   - In-batch hard negatives force the representation space to separate semantically adjacent clauses from the exact governing policy document.

2. **Overfitting & Generalization Trade-offs**:
   - Because the synthetic training corpus is focused (~21 training triplets), aggressive fine-tuning on a small sample risks catastrophic forgetting of general English queries.
   - Constraining the adapter via orthogonal projection preserves Euclidean norms and prevents embedding collapse.

3. **Reproduction Command**:
   ```bash
   python -m finetune.finetune_embeddings
   ```
