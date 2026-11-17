# DecisionIQ: System Architecture Specification

DecisionIQ is an enterprise-grade **Agentic RAG Assistant** designed for complex data-to-decision workflows in human resources, timesheet auditing, and payroll operations. It unifies unstructured policy documentation, structured relational business records, and unsupervised anomaly detection under a verified, self-correcting StateGraph.

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    User["Enterprise User / HR Business Partner"]
    UI["Streamlit Web UI (:8501)"]
    API["FastAPI Production Gateway (:8000)"]

    subgraph Guardrails["Enterprise Security & Guardrails Layer"]
        Shield["Prompt Injection Shield"]
        Scrubber["PII Scrubber (Regex & Named Entities)"]
        SQLGuard["AST SQL Guardrail (sqlglot)"]
    end

    subgraph AgentCore["LangGraph Agent Execution Engine"]
        Router["TF-IDF + LogisticRegression Intent Classifier"]
        Planner["Dynamic Multi-Step Tool Planner"]
        GroundingJudge["LLM Grounding Verification Judge"]
        QueryRewriter["Query Self-Correction Node"]
        Checkpointer["Session Memory Checkpointer (MemorySaver)"]
    end

    subgraph Tools["Tool Execution Suite"]
        DocRetriever["Hybrid Retrieval (Dense + BM25 RRF + Reranker)"]
        SQLTool["Safe Read-Only SQLite Analytics"]
        AnomalyTool["IsolationForest Timesheet Anomaly Detector"]
        Calculator["AST Safe Math Evaluator"]
    end

    subgraph Storage["Storage & Vector Infrastructure"]
        FAISS["FAISS Index (L2 Flat Index)"]
        Chroma["ChromaDB (HNSW Cosine Vector Store)"]
        SQLiteDB["SQLite Database (employees, timesheets, payroll)"]
        Artifacts["Joblib ML Models (Intent & Anomaly)"]
    end

    User --> UI
    UI --> API
    API --> Shield
    Shield --> Router
    Router --> Planner
    Planner --> Tools
    Tools --> GroundingJudge
    GroundingJudge -- "Ungrounded (Retry <= 2)" --> QueryRewriter
    QueryRewriter --> Planner
    GroundingJudge -- "Grounded" --> API
    GroundingJudge -- "Unanswerable / Max Retries Exceeded" --> API

    DocRetriever --> FAISS
    DocRetriever --> Chroma
    SQLTool --> SQLGuard
    SQLGuard --> SQLiteDB
    AnomalyTool --> Artifacts
    AnomalyTool --> SQLiteDB
    Router --> Artifacts
    AgentCore --> Checkpointer
    API --> Scrubber
```

---

## 2. LangGraph StateGraph Node Lifecycle & Conditional Edges

```mermaid
stateDiagram-v2
    [*] --> ClassifyIntent: User Query Received

    state ClassifyIntent {
        [*] --> MLClassifier: TF-IDF + LogisticRegression
        MLClassifier --> CheckConfidence: Confidence Threshold >= 0.28
        CheckConfidence --> RouteDecision: Confident
        CheckConfidence --> LLMFallback: Low Confidence (< 0.28)
        LLMFallback --> RouteDecision
    }

    ClassifyIntent --> Abstain: Intent == 'unanswerable' or 'adversarial'
    ClassifyIntent --> Planner: Intent in ['policy_rag', 'sql_analytics', 'hybrid_doc_sql', 'anomaly_detection']

    state Planner {
        [*] --> SelectTools: Generate Tool Sequence
    }

    Planner --> Tools: Execute Planned Tools

    state Tools {
        [*] --> ExecuteDocRetrieval: 'doc_retriever'
        [*] --> ExecuteSQLQuery: 'sql_query'
        [*] --> ExecuteAnomalyScan: 'detect_anomalies'
        [*] --> ExecuteMath: 'calculator'
    }

    Tools --> Synthesize: Gather Structured Evidence

    state Synthesize {
        [*] --> DraftAnswer: Inject Inline Citations [Doc: id] / [SQL: query]
    }

    Synthesize --> GroundingCheck: Verify Hallucination & Evidence

    state GroundingCheck {
        [*] --> LLMJudge: Compare Claims vs Retrieved Passages
    }

    GroundingCheck --> [*]: Verdict == 'GROUNDED'
    GroundingCheck --> QueryRewrite: Verdict == 'UNGROUNDED' and Retries < 2
    GroundingCheck --> Abstain: Verdict in ['ABSTAIN', 'UNGROUNDED'] and Retries >= 2

    state QueryRewrite {
        [*] --> Decontextualize: Reformulate Search Query
    }

    QueryRewrite --> Planner: Re-enter Planning Loop

    state Abstain {
        [*] --> RefusalResponse: 'I don't have enough verified information...'
    }

    Abstain --> [*]
```

---

## 3. End-to-End Query Execution Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as HR Specialist
    participant API as FastAPI Gateway
    participant Agent as LangGraph Agent
    participant ML as Scikit-Learn Router
    participant Retriever as Hybrid Retriever
    participant SQL as SafeSQLExecutor
    participant Judge as Grounding Judge

    User->>API: POST /api/v1/query (query="Overtime rule for L3 & Pune hours in March?")
    API->>Agent: invoke(query, thread_id)
    Agent->>ML: predict_intent_with_confidence(query)
    ML-->>Agent: ("hybrid_doc_sql", confidence=0.38)
    Agent->>Agent: planner: ["doc_retriever", "sql_query"]

    par Parallel Tool Execution
        Agent->>Retriever: retrieve("overtime rule L3", top_k=3)
        Retriever-->>Agent: [Doc: doc_overtime_2024#chunk_2]
    and
        Agent->>SQL: execute("SELECT SUM(overtime_hours)... WHERE location='Pune'")
        SQL-->>Agent: rows=[{"total_hours": 193.0}]
    end

    Agent->>Agent: synthesize: Draft response with inline citations
    Agent->>Judge: verify_grounding(answer, retrieved_evidence)
    Judge-->>Agent: {"verdict": "GROUNDED", "reasoning": "Clauses supported"}
    Agent-->>API: QueryResponse(answer, citations, grounded=True, latency_ms=68.1)
    API-->>User: JSON / SSE Token Stream
```

---

## 4. Relational Database Schema (`decisioniq.db`)

```mermaid
erDiagram
    employees ||--o{ timesheets : "logs daily"
    employees ||--o{ payroll_runs : "aggregated in"

    employees {
        int id PK
        string employee_code UK
        string name
        string email
        string department
        string location
        string grade
        float hourly_rate
        string hire_date
        string status
    }

    timesheets {
        int id PK
        int employee_id FK
        string date
        string project_code
        float regular_hours
        float overtime_hours
        string status
        int is_weekend
        int is_anomaly
        string anomaly_reason
    }

    payroll_runs {
        int id PK
        int month
        int year
        string location
        string department
        float total_regular_hours
        float total_overtime_hours
        float total_regular_pay
        float total_overtime_pay
    }

    user_feedback {
        string feedback_id PK
        string query_id
        int rating
        string comment
        string thread_id
        string created_at
    }
```

---

## 5. Dual Vector Store Abstraction Architecture

```mermaid
classDiagram
    class BaseVectorStore {
        <<abstract>>
        +add_documents(documents: List[Document]) List[str]
        +similarity_search(query: str, k: int) List[Document]
        +similarity_search_with_score(query: str, k: int) List[Tuple[Document, float]]
        +persist() void
        +count() int
    }

    class FAISSStore {
        -IndexFlatL2 index
        -Docstore docstore
        +add_documents()
        +similarity_search_with_score()
        +persist()
        +count()
    }

    class ChromaStore {
        -Client client
        -Collection collection
        +add_documents()
        +similarity_search_with_score()
        +persist()
        +count()
    }

    BaseVectorStore <|-- FAISSStore
    BaseVectorStore <|-- ChromaStore
```

---

## 6. Multi-Agent Supervisor Variant Architecture

```mermaid
flowchart LR
    UserQuery["User Business Query"]
    Supervisor["Supervisor Agent (Router & Aggregator)"]
    PolicyAgent["Policy Specialist Agent (RAG & Versioning)"]
    DataAgent["Data Specialist Agent (SQL & Anomaly Scan)"]
    CriticAgent["Critic Agent (Grounding & Citation Audit)"]
    FinalDecision["Synthesized Enterprise Decision"]

    UserQuery --> Supervisor
    Supervisor -- "Needs Policy Evidence" --> PolicyAgent
    PolicyAgent --> Supervisor
    Supervisor -- "Needs Records / Audits" --> DataAgent
    DataAgent --> Supervisor
    Supervisor -- "Specialists Complete" --> CriticAgent
    CriticAgent -- "Grounded & Verified" --> FinalDecision
    CriticAgent -- "Ungrounded / Hallucinated" --> Supervisor
```
