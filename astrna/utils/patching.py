from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any


@dataclass
class _WrapperState:
    target_ref: weakref.ReferenceType[Any] | None
    target_strong: Any
    original: Any
    active: bool


_WRAPPER_STATES: dict[int, _WrapperState] = {}


def same_callable(first: Any, second: Any) -> bool:
    """比较普通函数和绑定方法是否指向同一个可调用对象。"""
    if first is second:
        return True
    first_func = getattr(first, "__func__", first)
    second_func = getattr(second, "__func__", second)
    first_self = getattr(first, "__self__", None)
    second_self = getattr(second, "__self__", None)
    return first_func is second_func and first_self is second_self


def _split_callable(value: Any) -> tuple[Any, Any]:
    return getattr(value, "__func__", value), getattr(value, "__self__", None)


def _get_wrapper_state(wrapper: Any) -> _WrapperState | None:
    target, _ = _split_callable(wrapper)
    state = _WRAPPER_STATES.get(id(target))
    if state is None:
        return None
    if state.target_ref is not None:
        return state if state.target_ref() is target else None
    return state if state.target_strong is target else None


def mark_wrapper_active(wrapper: Any, original: Any) -> None:
    target, _ = _split_callable(wrapper)
    target_id = id(target)
    target_ref = None
    target_strong = None
    try:
        def discard_target(ref: weakref.ReferenceType[Any]) -> None:
            state = _WRAPPER_STATES.get(target_id)
            if state is not None and state.target_ref is ref:
                _WRAPPER_STATES.pop(target_id, None)

        target_ref = weakref.ref(target, discard_target)
    except TypeError:
        target_strong = target
    _WRAPPER_STATES[target_id] = _WrapperState(
        target_ref=target_ref,
        target_strong=target_strong,
        original=original,
        active=True,
    )
    try:
        target._astrna_wrapper_active = True
        target._astrna_wrapped_original = original
    except Exception:  # noqa: BLE001
        pass


def mark_wrapper_inactive(wrapper: Any) -> None:
    if wrapper is None:
        return
    state = _get_wrapper_state(wrapper)
    if state is not None:
        state.active = False
    target, _ = _split_callable(wrapper)
    try:
        target._astrna_wrapper_active = False
    except Exception:  # noqa: BLE001
        pass


def is_wrapper_active(wrapper: Any) -> bool:
    state = _get_wrapper_state(wrapper)
    return state is None or state.active


def unwrap_inactive_wrapper(func: Any) -> Any:
    seen: set[tuple[int, int]] = set()
    while callable(func):
        target, owner = _split_callable(func)
        identity = (id(target), id(owner) if owner is not None else 0)
        state = _get_wrapper_state(func)
        if state is None or state.active or identity in seen:
            break
        seen.add(identity)
        original = state.original
        if not callable(original) or same_callable(original, func):
            break
        func = original
    return func
