# Stage 4 SmartCredit Parser Hardening

## Background
In Stage 3 we introduced deterministic extractors (regex/token blocks) and telemetry for the SmartCredit report, with an optional legacy LLM path via `ENABLE_LLM_PARSING`. Stage 4 focuses on closing remaining gaps before moving to full rollout.

## Problem
- Parity vs. the legacy LLM parser has not been measured on a sufficiently diverse sample, especially for edge fields such as collections, charge‑off accounts, non‑standard dates, and payment histories.
- Formatting variance (spacing, Unicode, OCR artefacts) can break deterministic extraction.
- Fallback is coarse: a single field failure forces the whole report down the legacy LLM path.
- Telemetry exists but lacks dashboards, alerts, and SLOs.

## Tasks
### Parity & Coverage
- Run parity evaluation on 100–200 SmartCredit reports covering diverse formats.
- Metrics: ≥95% field‑level match for critical fields (accounts, amounts, statuses, dates) and ≥98% section detection.
- Produce a field‑wise diff report (by bureau and by account) and list outliers.

### Extractor Hardening
- Property‑based tests for dates (`MM/DD/YYYY`, `MMM YYYY`, `YYYY`) and numbers (currency symbols, commas, negatives).
- Light fuzzing for tokens/sections (whitespace, line breaks, Unicode, OCR quirks).
- Targeted tests for collections, charge‑off, student loans, mortgage, auto, joint, closed, and dispute cases.

### Smart Fallback
- If a single field fails deterministic extraction, log a warning and fall back to the legacy LLM for that field only.
- Ensure idempotency and consistent error codes.

### Observability
- Dashboard: section parse times, per‑field timings, coverage %, failure rate by cause.
- Alerts: SLOs for p95 parse time, failure rate, unknown‑field rate.
- PII masking/anonymisation in logs.

### Schema & Documentation
- Lock in the SmartCredit JSON schema (see below) covering Personal Info, Summary, Accounts, Public Info, Inquiries, and Meta/Scores.
- Short README: flags, flow, how to run tests/load tests, and how to switch parser backends.

## Acceptance Criteria
- Parity targets met with no open P1 issues.
- Tests cover key format variations; load test passes within budget.
- Field‑level fallback to LLM implemented and tested.
- Dashboard and alerts live with PII‑safe logs.
- JSON schema finalised and documented.

---

# SmartCredit Field Catalog
The following catalogue describes the hierarchical structure and field definitions used by the deterministic SmartCredit parser and OCR label hints.

## A) Meta & Scores (Header)
- **provider** — e.g. "SmartCredit / ConsumerDirect".  Labels: Provider, Source, Delivered By, SmartCredit.
- **model** — e.g. "VantageScore 3.0".  Labels: Model, Score Model, VantageScore.
- **report_view_timestamp** — timestamp printed in header/footer.  Labels: Report Viewed, Report Date/Time, Generated.
- **scores[]** (per bureau)
  - **bureau** — enum: TransUnion|Experian|Equifax.  Labels: TransUnion, Experian, Equifax.
  - **score** — integer (300–850 typically).  Labels: Score, Credit Score.
  - **as_of_date** — date the score applies to.  Labels: As of, Score Date.

*Normalization:* Dates → `YYYY-MM-DD` (accept `MM/DD/YYYY`, `MMM YYYY`, or just `YYYY`). Enums canonicalised.

## B) Personal Information (per bureau)
Fields repeated for each bureau (TransUnion, Experian, Equifax):
- **credit_report_date** — date the bureau’s credit report was pulled. Labels: Credit Report, Report Date, As of.
- **name** — full name. Labels: Name, Consumer Name.
- **also_known_as** — optional AKA names. Labels: Also Known As, AKA, Other Names.
- **date_of_birth** — year or full date. Labels: DOB, Date of Birth, Birth Year.
- **current_address** — most recent address block. Labels: Current Address, Present Address.
- **previous_addresses[]** — list of past addresses. Labels: Previous Address(es), Prior Address(es).
- **employer** — current/last employer (optional). Labels: Employer, Current Employer.
- **consumer_statement** — consumer statement or “None Reported”. Labels: Consumer Statement, Statement, Note.

*Normalization:* Addresses kept as raw multi‑line strings. "None Reported"/"No Data" ⇒ null.

## C) Summary (per bureau)
- **total_accounts** — int. Labels: Total Accounts, Accounts Total.
- **open_accounts** — int. Labels: Open Accounts, Open.
- **closed_accounts** — int. Labels: Closed Accounts, Closed.
- **delinquent_accounts** — int. Labels: Delinquent, Past Due Accounts.
- **derogatory_accounts** — int. Labels: Derogatory, Negative Accounts.
- **balances_total** — currency/float. Labels: Balances, Total Balances, Balance Total.
- **payments_total** — currency/float (monthly payments total). Labels: Payments, Total Payments, Monthly Payments.
- **public_records_count** — int. Labels: Public Records, Public Record(s).
- **inquiries_24m_count** — int. Labels: Inquiries (24 months), Inquiries Last 2 Years.

*Normalization:* Currency: strip $, commas → float. Missing/-- ⇒ null.

## D) Accounts
Accounts are grouped visually; keep `category_display` as seen:
- **Categories:** Revolving Accounts, Installment Accounts, Other Accounts, Collection Accounts (sometimes "Collections").

Per account (shared across bureaus):
- **creditor_name** — account header line. Labels: top block title, Creditor, Lender, Collections Agency.
- **category_display** — one of the 4 groups above.

Per account, per bureau (`per_bureau[]`):
- **bureau** — TransUnion|Experian|Equifax.
- **account_number_masked** — masked number (keep last4 if available). Labels: Account Number, Acct #, Acct No.
- **account_type** — high-level type (Credit Card, Auto Loan, Charge Account, Collection, Mortgage, Student Loan, Other). Labels: Type, Account Type.
- **account_type_detail** — optional detail. Labels: Type, Portfolio Type, Loan Type.
- **creditor_type** — bureau/industry classification. Labels: Creditor Type, Industry, Business Type.
- **account_status** — OPEN|CLOSED|PAID|DEROGATORY|COLLECTION|CHARGEOFF|SETTLED|... Labels: Account Status, Status.
- **payment_status** — CURRENT|LATE_30|LATE_60|LATE_90|LATE_120|COLLECTION|CHARGEOFF|... Labels: Payment Status, Pay Status.
- **account_rating** — high-level rating (often mirrors status). Labels: Rating, Account Rating.
- **account_description** — ownership/conditions (Individual|Joint|Terminated|Authorized User|...). Labels: Description, Ownership, Account Designator.
- **date_opened** — date. Labels: Date Opened, Opened.
- **closed_date** — date or null. Labels: Date Closed, Closed.
- **date_reported** — last reported date. Labels: Date Reported, Reported.
- **date_of_last_activity** — DLA if present. Labels: Date of Last Activity, DLA.
- **last_verified** — verification date if shown. Labels: Last Verified, Verified.
- **credit_limit** — for revolving; sometimes noted as "H/C".
- **high_balance** — highest balance / "H/C" depending on bureau.
- **balance_owed** — current balance. Labels: Balance, Current Balance, Amt Owed.
- **past_due_amount** — past due. Labels: Past Due, Amount Past Due.
- **payment_amount** — scheduled monthly payment. Labels: Monthly Payment, Payment.
- **last_payment** — date of last payment. Labels: Last Payment, Date of Last Payment.
- **term_length** — e.g., "54 Month(s)" for installment. Labels: Terms, Term, Months.
- **payment_frequency** — Monthly|--|null. Labels: Payment Frequency.
- **creditor_remarks** — free‑text remarks. Labels: Remarks, Comment, Note.
- **dispute_status** — account not disputed / dispute text. Labels: Dispute, Dispute Status.
- **original_creditor** — for collections. Labels: Original Creditor, Orig. Creditor.
- **two_year_payment_history[]** — monthly array of `{year, month, status}` with statuses like OK|30|60|90|120|CO|COLL|--. Labels: Payment History, 24-Month Payment History.
- **days_late_7y** — object `{late30, late60, late90}` across 7 years. Labels: Late 30/60/90 (7 years), Delinquencies (7y).

*Normalization & Enums:* Dates → `YYYY-MM-DD|YYYY-MM|YYYY`; money → float; `--` → null. Canonical enums:
- `ACCOUNT_STATUS`: OPEN|CLOSED|PAID|DEROGATORY|COLLECTION|CHARGEOFF|SETTLED|OTHER
- `PAYMENT_STATUS`: CURRENT|LATE_30|LATE_60|LATE_90|LATE_120|COLLECTION|CHARGEOFF|OTHER
- `ACCOUNT_TYPE`: CREDIT_CARD|AUTO_LOAN|CHARGE_ACCOUNT|MORTGAGE|STUDENT_LOAN|COLLECTION|OTHER

## E) Public Information
If present; otherwise list may be empty or “None Reported”.
- **record_type** — Bankruptcy|Lien|Judgment. Labels: Public Record, Record Type.
- **filing_date** — date. Labels: Filed, Filing Date.
- **status** — free text (e.g., Released, Discharged). Labels: Status.
- **court** — court name/locale (optional). Labels: Court, Court Name.
- **amount** — currency/float (optional). Labels: Amount, Liability.
- **reference_id** — docket/case/reference id (optional). Labels: Reference #, Case #, Docket #.

## F) Inquiries
Hard‑pull inquiries list (per line).
- **creditor_name** — inquirer. Labels: Creditor, Subscriber, Requester.
- **date_of_inquiry** — date. Labels: Date, Inquiry Date.
- **bureau** — where recorded (TransUnion|Experian|Equifax). Labels: bureau badges/columns.

## G) JSON Schema Skeleton
```json
{
  "provider": "SmartCredit",
  "model": "VantageScore 3.0",
  "report_view_timestamp": "string",
  "scores": [
    { "bureau": "TransUnion", "score": 0, "as_of_date": "YYYY-MM-DD" },
    { "bureau": "Experian", "score": 0, "as_of_date": "YYYY-MM-DD" },
    { "bureau": "Equifax", "score": 0, "as_of_date": "YYYY-MM-DD" }
  ],
  "personal_information": [
    {
      "bureau": "TransUnion",
      "credit_report_date": "YYYY-MM-DD",
      "name": "string",
      "also_known_as": "string|null",
      "date_of_birth": "YYYY|YYYY-MM-DD|null",
      "current_address": "string",
      "previous_addresses": ["string"],
      "employer": "string|null",
      "consumer_statement": "string|null"
    }
  ],
  "summary": [
    {
      "bureau": "TransUnion",
      "total_accounts": 0,
      "open_accounts": 0,
      "closed_accounts": 0,
      "delinquent_accounts": 0,
      "derogatory_accounts": 0,
      "balances_total": 0.0,
      "payments_total": 0.0,
      "public_records_count": 0,
      "inquiries_24m_count": 0
    }
  ],
  "accounts": [
    {
      "creditor_name": "string",
      "category_display": "Revolving|Installment|Other|Collection",
      "per_bureau": [
        {
          "bureau": "TransUnion",
          "account_number_masked": "string",
          "account_type": "CREDIT_CARD|AUTO_LOAN|CHARGE_ACCOUNT|MORTGAGE|STUDENT_LOAN|COLLECTION|OTHER",
          "account_type_detail": "string|null",
          "creditor_type": "string|null",
          "account_status": "OPEN|CLOSED|PAID|DEROGATORY|COLLECTION|CHARGEOFF|SETTLED|OTHER",
          "payment_status": "CURRENT|LATE_30|LATE_60|LATE_90|LATE_120|COLLECTION|CHARGEOFF|OTHER",
          "account_rating": "string|null",
          "account_description": "Individual|Joint|Authorized User|Terminated|...",
          "date_opened": "YYYY-MM-DD|null",
          "closed_date": "YYYY-MM-DD|null",
          "date_reported": "YYYY-MM-DD|null",
          "date_of_last_activity": "YYYY-MM-DD|null",
          "last_verified": "YYYY-MM-DD|null",
          "credit_limit": 0.0,
          "high_balance": 0.0,
          "balance_owed": 0.0,
          "past_due_amount": 0.0,
          "payment_amount": 0.0,
          "last_payment": "YYYY-MM-DD|null",
          "term_length": "string|null",
          "payment_frequency": "Monthly|--|null",
          "creditor_remarks": "string|null",
          "dispute_status": "string|null",
          "original_creditor": "string|null",
          "two_year_payment_history": [
            { "year": 2024, "month": "Oct", "status": "OK|30|60|90|120|CO|COLL|--" }
          ],
          "days_late_7y": { "late30": 0, "late60": 0, "late90": 0 }
        }
      ]
    }
  ],
  "public_information": {
    "items": [
      {
        "record_type": "Bankruptcy|Lien|Judgment",
        "filing_date": "YYYY-MM-DD",
        "status": "string",
        "court": "string|null",
        "amount": 0.0,
        "reference_id": "string|null"
      }
    ]
  },
  "inquiries": [
    { "creditor_name": "string", "date_of_inquiry": "YYYY-MM-DD", "bureau": "TransUnion" }
  ]
}
```

## H) OCR Label Hints
Common label variants to aid OCR:
- **Dates:** Date, Reported, As of, Opened, Closed, Verified, Last Activity, Last Payment, Filed.
- **Money:** Balance, High Balance, High Credit, Credit Limit, Past Due, Payment, Amount.
- **Statuses:** Account Status, Payment Status, Rating, Remarks, Dispute.
- **Parties:** Creditor, Original Creditor, Employer, Court.
- **Bureau markers:** TransUnion, Experian, Equifax.
