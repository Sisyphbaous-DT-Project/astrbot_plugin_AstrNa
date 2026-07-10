from __future__ import annotations

from functools import wraps
from types import MethodType

from astrna.utils.patching import (
    is_wrapper_active,
    mark_wrapper_active,
    mark_wrapper_inactive,
    same_callable,
    unwrap_inactive_wrapper,
)


def test_same_callable_compares_bound_method_owner_and_function():
    class Target:
        def method(self):
            return None

    first = Target()
    second = Target()

    assert same_callable(first.method, first.method)
    assert not same_callable(first.method, second.method)
    assert not same_callable(first.method, Target.method)


def test_bound_wrapper_state_is_visible_from_bare_function():
    class Target:
        pass

    target = Target()

    def original():
        return "original"

    def wrapper(self):
        return original()

    bound_wrapper = MethodType(wrapper, target)
    mark_wrapper_active(bound_wrapper, original)

    assert is_wrapper_active(wrapper)
    mark_wrapper_inactive(bound_wrapper)
    assert not is_wrapper_active(wrapper)
    assert unwrap_inactive_wrapper(bound_wrapper) is original


def test_functools_wraps_outer_is_not_owned_or_unwrapped():
    def original():
        return "original"

    def astrna_wrapper():
        return original()

    mark_wrapper_active(astrna_wrapper, original)

    @wraps(astrna_wrapper)
    def third_party_outer():
        return astrna_wrapper()

    mark_wrapper_inactive(astrna_wrapper)

    assert is_wrapper_active(third_party_outer)
    assert unwrap_inactive_wrapper(third_party_outer) is third_party_outer
    assert unwrap_inactive_wrapper(astrna_wrapper) is original
