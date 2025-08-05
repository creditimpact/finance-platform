# Constants and helpers used across the logic package.

# Allowed action tags used for account recommendations.
VALID_ACTION_TAGS = {
    "dispute",
    "goodwill",
    "custom_letter",
    "ignore",
}

# Map common strategist phrases to canonical action tags.
_ACTION_ALIAS_MAP = {
    "dispute": "dispute",
    "delete": "dispute",
    "remove": "dispute",
    "dispute_for_verification": "dispute",
    "challenge_the_debt": "dispute",
    "request_deletion": "dispute",
    "dispute_the_accuracy": "dispute",
    "verify_this_record": "dispute",
    "initiate_a_dispute": "dispute",
    "initiate_dispute": "dispute",
    "goodwill": "goodwill",
    "good_will": "goodwill",
    "goodwill_letter": "goodwill",
    "goodwill adjustment": "goodwill",
    "custom": "custom_letter",
    "custom_letter": "custom_letter",
    "custom letter": "custom_letter",
    "ignore": "ignore",
    "none": "ignore",
    "no_action": "ignore",
    "no action": "ignore",
    "monitor": "ignore",
}

_DISPLAY_NAME = {
    "dispute": "Dispute",
    "goodwill": "Goodwill",
    "custom_letter": "Custom Letter",
    "ignore": "Ignore",
}


def normalize_action_tag(raw: str | None) -> tuple[str, str]:
    """Return (action_tag, recommended_action) for a strategist value.

    The returned action_tag will be one of ``VALID_ACTION_TAGS`` or an
    empty string if the value is unrecognised. ``recommended_action`` is a
    human friendly label for display purposes.
    """
    if not raw:
        return "", ""
    key = str(raw).strip().lower().replace(" ", "_")
    tag = _ACTION_ALIAS_MAP.get(key)
    if not tag:
        return "", str(raw).strip()
    return tag, _DISPLAY_NAME.get(tag, tag.title())
