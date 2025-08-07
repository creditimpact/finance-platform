## Dev Notes

- Added strict warning policy (`-W error`) to pytest and resolved all existing warnings.
- Replaced deprecated `datetime.utcnow()` calls with timezone-aware `datetime.now(UTC)`.
- Adjusted tests to avoid or capture sanitization warnings and ensured files are closed.
