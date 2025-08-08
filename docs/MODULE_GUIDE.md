# Module Guide

| Module | Role | Public API | Key Dependencies |
| ------ | ---- | ---------- | ---------------- |
| `orchestrators.py` | Coordinates the end-to-end credit repair pipeline. | `process_client_intake`, `classify_client_responses`, `analyze_credit_report`, `generate_strategy_plan` | `logic.*`, `session_manager`, `analytics_tracker` |
| `models/` | Dataclasses representing accounts, bureaus, letters and strategy plans. | `Account`, `BureauAccount`, `BureauSection`, `LetterAccount`, `LetterContext`, `LetterArtifact`, `Recommendation`, `StrategyItem`, `StrategyPlan` | `dataclasses`, `typing` |
| `logic/` | Business logic: report parsing, strategy generation, compliance checks and PDF rendering. | `analyze_credit_report`, `StrategyGenerator`, `run_compliance_pipeline`, `pdf_ops.convert_txts_to_pdfs` | OpenAI API, PDF utilities |
| `services/` | Lightweight wrappers for external integrations. | `AIClient`, email utilities | `requests`, `smtplib` |
| `templates/` | Jinja2 letter templates rendered into HTML/PDF. | N/A – consumed by letter generation code | `Jinja2`, `logic.utils.pdf_ops` |
