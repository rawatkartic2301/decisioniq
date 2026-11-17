"""Synthetic instruction-tuning dataset generator for domain LoRA / PEFT fine-tuning.

Produces structured instruction, input, output triplets formatted for SFTTrainer
covering (1) citation-grounded policy Q&A, (2) Text-to-SQL synthesis, and (3) grounding judge critiques.
"""

import json
import logging
from pathlib import Path
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOMAIN_SFT_EXAMPLES = [
    # Category 1: Policy Q&A with Citation Formatting
    {
        "instruction": "You are DecisionIQ, an enterprise assistant. Answer the policy question using the provided context, citing sources in the exact format [Doc: <source_doc>#<chunk_id>]. If the information is not in the context, explicitly state that you don't have enough verified information.",
        "input": "Context: [Doc: hr_leave_policy_v2_2024.md#doc_leave_2024#chunk_1]: Privilege Leave (PL): All full-time regular employees accrue 1.75 days of PL per completed calendar month of service, totaling 21 days per calendar year. Maximum carry-forward to the subsequent calendar year is capped at 45 days. Any accrued balance exceeding 45 days as of December 31 lapses automatically.\n\nQuestion: What is the maximum number of privilege leaves that an employee can carry forward?",
        "output": "Based on the official enterprise policy [Doc: hr_leave_policy_v2_2024.md#doc_leave_2024#chunk_1], full-time regular employees may carry forward a maximum of 45 privilege leave days into the subsequent calendar year. Any balance exceeding 45 days as of December 31 lapses automatically.",
    },
    {
        "instruction": "You are DecisionIQ, an enterprise assistant. Answer the policy question using the provided context, citing sources in the exact format [Doc: <source_doc>#<chunk_id>]. If the information is not in the context, explicitly state that you don't have enough verified information.",
        "input": "Context: [Doc: overtime_policy_l1_l5.md#doc_overtime_2024#chunk_2]: Grade L3 (Specialist / Lead): Eligible for overtime ONLY with advance manager pre-approval. Overtime multiplier is 1.5x regular hourly rate for both weekday and weekend hours. Holiday hours are compensated at 2.0x.\n\nQuestion: What is the overtime rule for grade L3 employees?",
        "output": "According to the overtime policy [Doc: overtime_policy_l1_l5.md#doc_overtime_2024#chunk_2], Grade L3 employees are eligible for overtime only with advance manager pre-approval. Overtime hours are compensated at 1.5x the regular hourly rate for both weekdays and weekends, and 2.0x for designated company holidays.",
    },
    {
        "instruction": "You are DecisionIQ, an enterprise assistant. Answer the policy question using the provided context, citing sources in the exact format [Doc: <source_doc>#<chunk_id>]. If the information is not in the context, explicitly state that you don't have enough verified information.",
        "input": "Context: [Doc: it_security_policy.md#doc_it_sec#chunk_1]: USB Storage and External Drives: Employees are strictly prohibited from connecting unauthorized personal USB drives, portable hard drives, or external storage media to company laptops.\n\nQuestion: Does the company reimburse pet insurance expenses?",
        "output": "I don't have enough verified information in the official policy documents to answer this question. The provided context does not contain policy provisions regarding pet insurance reimbursement.",
    },
    # Category 2: Safe Text-to-SQL Generation
    {
        "instruction": "Generate a valid, read-only SQLite SELECT query for the DecisionIQ database. Tables available: employees(id, name, department, location, grade, hourly_rate, status), timesheets(id, employee_id, date, regular_hours, overtime_hours, status, is_weekend, is_anomaly), payroll_runs(id, month, year, location, department, total_regular_hours, total_overtime_hours, total_regular_pay). Always apply a LIMIT clause if not aggregating.",
        "input": "How many active employees work in the Engineering department in Pune?",
        "output": "SELECT COUNT(*) AS active_engineering_count FROM employees WHERE department = 'Engineering' AND location = 'Pune' AND status = 'active';",
    },
    {
        "instruction": "Generate a valid, read-only SQLite SELECT query for the DecisionIQ database. Tables available: employees(id, name, department, location, grade, hourly_rate, status), timesheets(id, employee_id, date, regular_hours, overtime_hours, status, is_weekend, is_anomaly), payroll_runs(id, month, year, location, department, total_regular_hours, total_overtime_hours, total_regular_pay). Always apply a LIMIT clause if not aggregating.",
        "input": "How many total overtime hours were logged by the Pune location in March 2024?",
        "output": "SELECT SUM(t.overtime_hours) AS total_march_overtime FROM timesheets t JOIN employees e ON t.employee_id = e.id WHERE e.location = 'Pune' AND t.date >= '2024-03-01' AND t.date <= '2024-03-31';",
    },
    {
        "instruction": "Generate a valid, read-only SQLite SELECT query for the DecisionIQ database. Tables available: employees(id, name, department, location, grade, hourly_rate, status), timesheets(id, employee_id, date, regular_hours, overtime_hours, status, is_weekend, is_anomaly), payroll_runs(id, month, year, location, department, total_regular_hours, total_overtime_hours, total_regular_pay). Always apply a LIMIT clause if not aggregating.",
        "input": "List the top 5 employees with the highest hourly rates in grade L5.",
        "output": "SELECT name, department, location, hourly_rate FROM employees WHERE grade = 'L5' ORDER BY hourly_rate DESC LIMIT 5;",
    },
    # Category 3: Grounding Judge Evaluation
    {
        "instruction": "You are a Grounding Critic. Evaluate whether the candidate answer is completely supported by the provided context. Output JSON with verdict ('GROUNDED', 'UNGROUNDED', or 'ABSTAIN') and reasoning.",
        "input": "Context: Standard per diem for London is GBP 120 per night. Expense claims must be submitted within 30 calendar days.\nClaim: Employees traveling to London receive GBP 120 per night and can submit claims within 60 days.",
        "output": '{"verdict": "UNGROUNDED", "reasoning": "The claim states expense claims can be submitted within 60 days, but the context explicitly restricts submission to 30 calendar days."}',
    },
    {
        "instruction": "You are a Grounding Critic. Evaluate whether the candidate answer is completely supported by the provided context. Output JSON with verdict ('GROUNDED', 'UNGROUNDED', or 'ABSTAIN') and reasoning.",
        "input": "Context: Maternity leave entitlement is 26 calendar weeks of fully paid leave for female employees with at least 80 days of continuous service.\nClaim: Female employees with 80+ days continuous service are eligible for 26 weeks paid maternity leave.",
        "output": '{"verdict": "GROUNDED", "reasoning": "All statements in the claim are directly verified and supported by the context."}',
    },
]


def generate_sft_dataset(
    output_dir: Path = Path("finetune/data"),
    train_ratio: float = 0.8,
) -> Tuple[int, int]:
    """Saves SFT instruction dataset in train and validation JSONL splits."""
    output_dir.mkdir(parents=True, exist_ok=True)

    split_idx = int(len(DOMAIN_SFT_EXAMPLES) * train_ratio)
    train_examples = DOMAIN_SFT_EXAMPLES[:split_idx]
    val_examples = DOMAIN_SFT_EXAMPLES[split_idx:]

    train_path = output_dir / "sft_train.jsonl"
    val_path = output_dir / "sft_val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    logger.info("Generated %d SFT training examples -> %s", len(train_examples), train_path)
    logger.info("Generated %d SFT validation examples -> %s", len(val_examples), val_path)
    return len(train_examples), len(val_examples)


if __name__ == "__main__":
    generate_sft_dataset()
