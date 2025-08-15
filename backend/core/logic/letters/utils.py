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
