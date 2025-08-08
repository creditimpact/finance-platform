"""Helpers for merging strategist outputs with bureau data."""

from __future__ import annotations

import re
from typing import Optional

import warnings
from typing import Iterable

from logic.constants import (
    FallbackReason,
    StrategistFailureReason,
    normalize_action_tag,
)
from logic.fallback_manager import determine_fallback_action
from logic.utils.names_normalization import normalize_creditor_name
from models.account import Account
from models.strategy import StrategyPlan


def merge_strategy_outputs(
    strategy_obj: StrategyPlan | dict, bureau_data_obj: dict[str, dict[str, list[Account | dict]]]
) -> None:
    """Align strategist ``strategy_obj`` with ``bureau_data_obj``.

    Parameters
    ----------
    strategy_obj:
        Strategy plan or equivalent dictionary.
    bureau_data_obj:
        Mapping of bureau -> section -> list of accounts.
    """

    if isinstance(strategy_obj, dict):
        try:
            warnings.warn(
                "dict strategy_obj is deprecated", DeprecationWarning, stacklevel=2
            )
        except DeprecationWarning:
            pass
        plan = StrategyPlan.from_dict(strategy_obj)
    else:
        plan = strategy_obj

    def norm_key(name: str, number: str) -> tuple[str, str]:
        norm_name = normalize_creditor_name(name)
        digits = re.sub(r"\D", "", number or "")
        last4 = digits[-4:] if digits else ""
        return norm_name, last4

    index = {}
    for item in plan.accounts:
        key = norm_key(item.name, item.account_number or "")
        index[key] = item

    for payload in bureau_data_obj.values():
        for items in payload.values():
            if not isinstance(items, list):
                continue
            for i, acc in enumerate(items):
                if isinstance(acc, dict):
                    try:
                        warnings.warn(
                            "dict account is deprecated", DeprecationWarning, stacklevel=2
                        )
                    except DeprecationWarning:
                        pass
                    acc_obj = Account.from_dict(acc)
                else:
                    acc_obj = acc

                key = norm_key(acc_obj.name, acc_obj.account_number or "")
                src = index.get(key)
                raw_action: Optional[str] = None
                if src is None:
                    acc_obj.extras["strategist_failure_reason"] = (
                        StrategistFailureReason.MISSING_INPUT
                    )
                    if isinstance(acc, dict):
                        items[i] = acc_obj.to_dict()
                    continue

                rec = src.recommendation
                raw_action = (
                    rec.recommended_action if rec else None
                ) or acc_obj.extras.get("recommendation")
                tag, action = normalize_action_tag(raw_action)
                if raw_action is None:
                    acc_obj.extras["strategist_failure_reason"] = (
                        StrategistFailureReason.EMPTY_OUTPUT
                    )
                elif raw_action and not tag:
                    acc_obj.extras["strategist_failure_reason"] = (
                        StrategistFailureReason.UNRECOGNIZED_FORMAT
                    )
                    acc_obj.extras["fallback_unrecognized_action"] = True
                if tag:
                    acc_obj.action_tag = tag
                    acc_obj.recommended_action = action
                elif raw_action:
                    acc_obj.recommended_action = raw_action

                acc_obj.extras["strategist_raw_action"] = raw_action
                if rec and rec.advisor_comment:
                    acc_obj.advisor_comment = rec.advisor_comment
                if rec and rec.flags:
                    acc_obj.flags = rec.flags

                if isinstance(acc, dict):
                    items[i] = acc_obj.to_dict()


def handle_strategy_fallbacks(
    bureau_data_obj: dict,
    classification_map: dict,
    audit=None,
    log_list: list | None = None,
) -> None:
    """Apply fallback logic and log strategy decisions."""

    for bureau, payload in bureau_data_obj.items():
        for section, items in payload.items():
            if not isinstance(items, list):
                continue
            for acc in items:
                raw_action = acc.get("strategist_raw_action")
                tag = acc.get("action_tag")
                failure_reason = acc.get("strategist_failure_reason")
                acc_id = acc.get("account_id") or acc.get("name")

                if failure_reason and audit:
                    audit.log_account(
                        acc_id,
                        {
                            "stage": "strategist_failure",
                            "failure_reason": failure_reason.value,
                            **(
                                {"raw_action": raw_action}
                                if (
                                    failure_reason
                                    == StrategistFailureReason.UNRECOGNIZED_FORMAT
                                    and raw_action
                                )
                                else {}
                            ),
                        },
                    )
                if failure_reason == StrategistFailureReason.MISSING_INPUT and log_list is not None:
                    log_list.append(
                        f"[{bureau}] No strategist entry for '{acc.get('name')}' ({acc.get('account_number')})"
                    )
                if (
                    failure_reason == StrategistFailureReason.UNRECOGNIZED_FORMAT
                    and raw_action
                ):
                    print(
                        f"[⚠️] Unrecognised strategist action '{raw_action}' for {acc.get('name')}"
                    )

                if not tag:
                    strategist_action = raw_action if raw_action else None
                    if raw_action is None:
                        fallback_reason = FallbackReason.NO_RECOMMENDATION
                    else:
                        raw_key = str(raw_action).strip().lower().replace(" ", "_")
                        fallback_reason = (
                            FallbackReason.KEYWORD_MATCH
                            if raw_key == FallbackReason.KEYWORD_MATCH.value
                            else FallbackReason.UNRECOGNIZED_TAG
                        )

                    fallback_action = determine_fallback_action(acc)
                    keywords_trigger = fallback_action == "dispute"

                    if keywords_trigger:
                        acc["action_tag"] = "dispute"
                        if raw_action:
                            acc["recommended_action"] = "Dispute"
                        else:
                            acc.setdefault("recommended_action", "Dispute")

                        if log_list is not None and (raw_action is None or not tag):
                            if raw_action:
                                log_list.append(
                                    f"[{bureau}] Fallback dispute overriding '{raw_action}' for '{acc.get('name')}' ({acc.get('account_number')})"
                                )
                            else:
                                log_list.append(
                                    f"[{bureau}] Fallback dispute (no recommendation) for '{acc.get('name')}' ({acc.get('account_number')})"
                                )
                    else:
                        if log_list is not None and (raw_action is None or not tag):
                            log_list.append(
                                f"[{bureau}] Evaluated fallback for '{acc.get('name')}' ({acc.get('account_number')})"
                            )

                    overrode_strategist = bool(raw_action) and bool(keywords_trigger)

                    if audit:
                        audit.log_account(
                            acc_id,
                            {
                                "stage": "strategy_fallback",
                                "fallback_reason": fallback_reason.value,
                                "strategist_action": strategist_action,
                                **(
                                    {"raw_action": strategist_action}
                                    if acc.get("fallback_unrecognized_action")
                                    and strategist_action
                                    else {}
                                ),
                                "overrode_strategist": overrode_strategist,
                                **(
                                    {"failure_reason": failure_reason.value}
                                    if failure_reason
                                    else {}
                                ),
                            },
                        )
                if audit:
                    cls = classification_map.get(str(acc.get("account_id")))
                    audit.log_account(
                        acc_id,
                        {
                            "stage": "strategy_decision",
                            "action": acc.get("action_tag") or None,
                            "recommended_action": acc.get("recommended_action"),
                            "flags": acc.get("flags"),
                            "reason": acc.get("advisor_comment")
                            or acc.get("analysis")
                            or raw_action,
                            "classification": cls,
                        },
                    )


def merge_strategy_data(
    strategy_obj: dict,
    bureau_data_obj: dict,
    classification_map: dict,
    audit=None,
    log_list: list | None = None,
) -> None:
    """Wrapper combining merge and fallback handling."""

    merge_strategy_outputs(strategy_obj, bureau_data_obj)
    handle_strategy_fallbacks(bureau_data_obj, classification_map, audit, log_list)


__all__ = [
    "merge_strategy_outputs",
    "handle_strategy_fallbacks",
    "merge_strategy_data",
]
