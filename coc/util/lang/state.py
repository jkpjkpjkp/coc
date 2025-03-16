from typing_extensions import Annotated, TypedDict


def add_raw(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right

def keep_last(left, right):
    if right:
        return right
    else:
        return left


class KPState(TypedDict):
    knowledge: Annotated[list[str], add_raw]
    plan: Annotated[str, keep_last]
    raw_data: Annotated[list[tuple[str]], add_raw]