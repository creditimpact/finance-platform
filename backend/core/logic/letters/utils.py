from __future__ import annotations

from typing import Any, Iterable, Mapping

from backend.core.models.account import Account


class StrategyContextMissing(RuntimeError):
    """Raised when an account lacks required strategy context."""

    def __init__(self, account_id: str | None):
        self.account_id = account_id
        super().__init__(f"Strategy context missing for account {account_id}")


def _get_fields(acc: Account | Mapping[str, Any]) -> tuple[str | None, str | None]:
    if isinstance(acc, Account):
        return acc.action_tag, acc.account_id
    if isinstance(acc, Mapping):
        return acc.get("action_tag"), acc.get("account_id")
    return getattr(acc, "action_tag", None), getattr(acc, "account_id", None)


def ensure_strategy_context(
    accounts: Iterable[Account | Mapping[str, Any]],
    enforcement_enabled: bool,
) -> None:
    """Ensure each account has an action tag when enforcement is enabled."""

    if not enforcement_enabled:
        return

    for acc in accounts:
        action_tag, account_id = _get_fields(acc)
        if action_tag:
            continue
        raise StrategyContextMissing(account_id)


def populate_required_fields(
    account: dict[str, Any], strat: Mapping[str, Any] | None = None
) -> None:
    """Fill per-action required fields using account and strategy data.

    The router validates that certain fields are present before rendering the
    final letter.  This helper ensures those fields are populated after strategy
    data has been merged into the account context.
    """

    tag = str(account.get("action_tag") or "").lower()

    # Basic collectors/furnishers -------------------------------------------------
    if tag in {"debt_validation", "pay_for_delete", "cease_and_desist"}:
        account.setdefault("collector_name", account.get("name"))

    if tag == "direct_dispute":
        account.setdefault("furnisher_address", account.get("address"))

    # Strategy‑provided fields -----------------------------------------------------
    strat = strat or {}
    if tag == "pay_for_delete" and strat.get("offer_terms") is not None:
        account.setdefault("offer_terms", strat.get("offer_terms"))

    if tag == "goodwill":
        for field in ["months_since_last_late", "account_history_good"]:
            if strat.get(field) is not None and not account.get(field):
                account[field] = strat[field]

    if tag == "mov":
        for field in ["cra_last_result", "days_since_cra_result"]:
            if strat.get(field) is not None and not account.get(field):
                account[field] = strat[field]


__all__ = [
    "StrategyContextMissing",
    "ensure_strategy_context",
    "populate_required_fields",
]
