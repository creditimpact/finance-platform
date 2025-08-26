from dataclasses import dataclass


@dataclass
class CaseStoreError(Exception):
    code: str
    message: str


# Known error codes
NOT_FOUND = "NOT_FOUND"
VALIDATION_FAILED = "VALIDATION_FAILED"
IO_ERROR = "IO_ERROR"


__all__ = ["CaseStoreError", "NOT_FOUND", "VALIDATION_FAILED", "IO_ERROR"]
