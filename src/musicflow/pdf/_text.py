def _numbers(values: list[int], limit: int = 8) -> str:
    shown = ", ".join(str(value) for value in values[:limit])
    return shown if len(values) <= limit else f"{shown}, ... ({len(values)} total)"
