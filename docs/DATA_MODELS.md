# Data Models

This document summarizes dataclasses under `models/` and their relationships.

## account.py

### `LateHistory`
- `date: str`
- `status: str`

### `Inquiry`
- `creditor_name: str`
- `date: str`
- `bureau: Optional[str]`

### `Account`
- `account_id: Optional[str]`
- `name: str`
- `account_number: Optional[str]`
- `reported_status: Optional[str]`
- `status: Optional[str]`
- `dispute_type: Optional[str]`
- `advisor_comment: Optional[str]`
- `action_tag: Optional[str]`
- `recommended_action: Optional[str]`
- `flags: List[str]`
- `extras: Dict[str, object]`

## bureau.py

### `BureauAccount`
- extends `Account`
- `bureau: Optional[str]`
- `section: Optional[str]`

### `BureauSection`
- `name: str`
- `accounts: List[BureauAccount]`

## letter.py

### `LetterAccount`
- `name: str`
- `account_number: str`
- `status: str`
- `paragraph: Optional[str]`
- `requested_action: Optional[str]`
- `personal_note: Optional[str]`

### `LetterContext`
- `client_name: str`
- `client_address_lines: List[str]`
- `bureau_name: str`
- `bureau_address: str`
- `date: str`
- `opening_paragraph: str`
- `accounts: List[LetterAccount]`
- `inquiries: List[Inquiry]`
- `closing_paragraph: str`
- `is_identity_theft: bool`

### `LetterArtifact`
- `html: str`
- `pdf_path: Optional[str]`

## strategy.py

### `Recommendation`
- `action_tag: Optional[str]`
- `recommended_action: Optional[str]`
- `advisor_comment: Optional[str]`
- `flags: List[str]`

### `StrategyItem`
- `account_id: str`
- `name: str`
- `account_number: Optional[str]`
- `recommendation: Recommendation | None`

### `StrategyPlan`
- `accounts: List[StrategyItem]`

All models expose `from_dict()` and `to_dict()` helpers for conversion to and from plain dictionaries.
