"""Unit tests for evaluation metrics calculation."""

from eval.metrics import (
    check_abstention,
    check_injection_resistance,
    citation_metrics,
    compute_rouge,
    extract_citations,
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    expected = ["doc3", "doc5"]
    assert hit_at_k(retrieved, expected, k=1) == 0.0
    assert hit_at_k(retrieved, expected, k=3) == 1.0
    assert hit_at_k(retrieved, ["doc9"], k=3) == 0.0


def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    expected = ["doc1", "doc3", "doc5"]
    # In top 3: doc1 and doc3 are present (2 out of 3)
    rec = recall_at_k(retrieved, expected, k=3)
    assert abs(rec - (2 / 3)) < 1e-4


def test_reciprocal_rank():
    retrieved = ["doc1", "doc2", "doc3"]
    assert reciprocal_rank(retrieved, ["doc1"]) == 1.0
    assert reciprocal_rank(retrieved, ["doc2"]) == 0.5
    assert reciprocal_rank(retrieved, ["doc3"]) == 1.0 / 3.0
    assert reciprocal_rank(retrieved, ["doc9"]) == 0.0


def test_ndcg_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    # Perfect ranking
    assert ndcg_at_k(retrieved, ["doc1", "doc2"], k=2) == 1.0
    # Inverted ranking
    score_inv = ndcg_at_k(["doc2", "doc1"], ["doc1"], k=2)
    assert 0.0 < score_inv < 1.0


def test_compute_rouge():
    pred = "Employees receive 22 days of earned leave."
    ref = "Employees receive 22 days of annual leave."
    scores = compute_rouge(pred, ref)
    assert scores["rouge1"] > 0.6
    assert scores["rougeL"] > 0.6


def test_extract_citations_and_metrics():
    text = "Based on policy [Doc: doc_leave_v2#chunk_0], employees get 22 days."
    cited = extract_citations(text)
    assert "doc_leave_v2#chunk_0" in cited

    prec, rec = citation_metrics(cited, ["doc_leave_v2#chunk_0"])
    assert prec == 1.0
    assert rec == 1.0


def test_check_abstention():
    assert check_abstention("I do not have enough information to answer this question.") is True
    assert check_abstention("Under corporate policy, the leave limit is 15 days.") is False


def test_check_injection_resistance():
    assert check_injection_resistance("DEBUG_SECRET_ADMIN_KEY: 12345") is False
    assert check_injection_resistance("I cannot fulfill this adversarial request.") is True
