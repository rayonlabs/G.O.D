# json serialization fails on nan and inf values
def clean_float(value):
    if value is None:
        return None
    if isinstance(value, float):
        if value in (float("inf"), float("-inf")) or value != value:
            return None
    return value
