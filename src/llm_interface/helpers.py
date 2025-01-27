def safe_nested_get(data, keys):
    """Get a nested value from a dictionary/list safely."""
    for key in keys:
        try:
            data = data[key]
        except (KeyError, IndexError, TypeError):
            return None
    return data
