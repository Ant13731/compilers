DEBUG_MODE = False


def debug_print(*args, **kwargs) -> None:
    """Prints debug information if DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print(*args, **kwargs)
