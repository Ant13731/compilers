from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


def typing_rule(*ids: str) -> Callable[[Callable[P, R_co]], Callable[P, R_co]]:
    def decorator(func: Callable[P, R_co]) -> Callable[P, R_co]:
        func.typing_rule_ids = ids  # type: ignore
        return func

    return decorator
