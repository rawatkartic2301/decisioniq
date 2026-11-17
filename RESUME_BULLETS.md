# DecisionIQ: High-Impact Resume Bullets

Use these tailored resume bullets for Senior AI Engineer / LLM Engineer applications (RAG, agentic systems, evaluation, fine-tuning). Every metric is **empirically measured and reproducible**.

---

## 💼 Core Resume Bullets (Tailored for Senior AI / LLM Engineer Roles)

* **Architected DecisionIQ, an Enterprise Agentic RAG assistant** combining unstructured HR/payroll policies and relational SQLite databases using **LangGraph**, achieving **100% abstention accuracy** on unanswerable queries and **100% prompt injection resistance** across a 75-item golden evaluation benchmark.

* **Engineered a hybrid retrieval engine** merging dense semantic vectors (`all-MiniLM-L6-v2`) and sparse BM25 Okapi search via **Reciprocal Rank Fusion (RRF)** and Cross-Encoder reranking behind an abstract dual-store interface (`FAISS` and `ChromaDB`), delivering **68.1ms p50 latency** on CPU-only execution.

* **Developed an automated grounding self-correction loop** in LangGraph with an LLM-as-judge that audits synthesized claims against retrieved chunk IDs, implementing bounded query rewriting ($N=2$) and inline source attribution (`[Doc: <chunk_id>]` and `[SQL: <query>]`).

* **Trained and deployed classical scikit-learn models**, including a TF-IDF + Logistic Regression intent router (**0.869 Macro F1**, **1.000 ROC-AUC** across 6 classes with sub-millisecond inference and LLM fallback) and an unsupervised `IsolationForest` detecting tabular timesheet anomalies (**66.2% F1**, **69.2% recall** on 4,000+ records).

* **Implemented AST-level SQL security guardrails** with `sqlglot` to parse and validate dynamic text-to-SQL queries, enforcing strict `SELECT`-only operations, table allowlists, forced `LIMIT 100`, read-only database connections, and a 5-second timeout, completely eliminating SQL injection vectors.

* **Built a production serving stack** with **FastAPI** (SSE streaming, request ID tracing, rate limiting, PII redaction) and an interactive **Streamlit** web UI displaying real-time agent decision traces, citations, and feedback collection, fully containerized via multi-stage **Docker Compose**.

---

## 🛠️ Tech Stack Keywords for ATS (Applicant Tracking Systems)

`LangGraph` • `LangChain` • `Agentic RAG` • `FAISS` • `ChromaDB` • `Hybrid Retrieval (RRF)` • `Cross-Encoder Reranking` • `Sentence-Transformers` • `Scikit-Learn` • `IsolationForest` • `AST SQL Guardrails (sqlglot)` • `LLM Grounding Verification` • `PEFT / LoRA` • `FastAPI` • `Streamlit` • `Docker Compose` • `MLflow` • `Pytest` • `Python 3.11`
