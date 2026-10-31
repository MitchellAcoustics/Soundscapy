"""Status: normative codes, ordering, counts, text columns and invariants."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from soundscapy._model import COL_FLAGS, COL_INCLUDED, COL_REASONS
from soundscapy.survey.status import Flag, Reason, Status


def test_reason_codes_normative_order():
    assert [r.value for r in Reason] == [
        "scale_violation", "all_identical", "excessive_missing", "no_consent",
        "test_response", "under_age", "paper_input", "incomplete_attributes",
        "uncovered_language", "low_confidence_language",
    ]  # fmt: skip


def test_flag_codes_normative_order():
    assert [f.value for f in Flag] == [
        "all_midpoint", "under_age", "paper_input", "medium_confidence_language",
        "equal_angles_fallback",
    ]  # fmt: skip


def test_with_reason_twice_keeps_one_copy():
    s = Status.all_included(2).with_reason([True, False], Reason.NO_CONSENT)
    s = s.with_reason([True, True], Reason.NO_CONSENT)
    assert s.reasons == ((Reason.NO_CONSENT,), (Reason.NO_CONSENT,))
    assert s.included.tolist() == [False, False]


def test_out_of_order_reasons_stored_in_enum_order():
    s = Status.all_included(1)
    s = s.with_reason([True], Reason.ALL_IDENTICAL).with_reason([True], Reason.SCALE_VIOLATION)
    assert s.reasons == ((Reason.SCALE_VIOLATION, Reason.ALL_IDENTICAL),)


def test_with_methods_return_new_objects():
    base = Status.all_included(2)
    changed = base.with_reason([True, False], Reason.NO_CONSENT).with_flag(
        [False, True], Flag.ALL_MIDPOINT
    )
    assert base.included.all()
    assert base.reasons == ((), ())
    assert base.flags == ((), ())
    assert changed.flags == ((), (Flag.ALL_MIDPOINT,))


def test_label_arrays_and_frame_text():
    s = Status.all_included(2)
    s = s.with_reason([True, False], Reason.ALL_IDENTICAL).with_reason(
        [True, False], Reason.SCALE_VIOLATION
    )
    s = s.with_flag([False, True], Flag.ALL_MIDPOINT)
    arrays = s.label_arrays()
    assert arrays[COL_INCLUDED].dtype == np.bool_
    assert arrays[COL_REASONS].tolist() == ["scale_violation;all_identical", ""]
    assert arrays[COL_FLAGS].tolist() == ["", "all_midpoint"]
    frame = s.to_frame()
    assert frame[COL_REASONS].tolist() == ["scale_violation;all_identical", ""]
    assert frame[COL_FLAGS].tolist() == ["", "all_midpoint"]
    assert frame[COL_INCLUDED].tolist() == [False, True]


def test_counts_all_included():
    counts = Status.all_included(3).counts()
    assert counts["n"] == 3
    assert counts["included"] == 3
    assert counts["excluded"] == 0
    assert set(counts) == {
        "n", "included", "excluded",
        *(f"reason:{r.value}" for r in Reason), *(f"flag:{f.value}" for f in Flag),
    }  # fmt: skip
    assert all(v == 0 for k, v in counts.items() if ":" in k)


def test_counts_reason_and_flag_kept_apart():
    s = Status.all_included(3)
    s = s.with_reason([True, False, False], Reason.UNDER_AGE).with_flag(
        [False, True, False], Flag.UNDER_AGE
    )
    counts = s.counts()
    assert counts["reason:under_age"] == 1
    assert counts["flag:under_age"] == 1
    assert counts["excluded"] == 1


def test_arrays_read_only():
    s = Status.all_included(2)
    with pytest.raises(ValueError, match="read-only"):
        s.included[0] = False
    for array in s.label_arrays().values():
        assert not array.flags.writeable


def test_constructor_copies_included():
    source = np.ones(1, dtype=np.bool_)
    s = Status(source, ((),), ((),))
    source[0] = False
    assert s.included[0]


def test_mismatched_included_raises():
    with pytest.raises(ValueError, match="exactly where"):
        Status(np.array([True]), ((Reason.NO_CONSENT,),), ((),))


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        Status(np.array([True, True]), ((),), ((),))


def test_unordered_or_repeated_codes_raise():
    out_of_order = (Reason.ALL_IDENTICAL, Reason.SCALE_VIOLATION)
    with pytest.raises(ValueError, match="enum order"):
        Status(np.array([False]), (out_of_order,), ((),))
    with pytest.raises(ValueError, match="enum order"):
        Status(np.array([True]), ((),), ((Flag.ALL_MIDPOINT, Flag.ALL_MIDPOINT),))


def test_foreign_code_raises():
    with pytest.raises(ValueError, match="enum order"):
        Status(np.array([False]), (("not_a_reason",),), ((),))  # ty: ignore[invalid-argument-type]  # deliberate: exercises the runtime ValueError


def test_bad_mask_raises():
    with pytest.raises(ValueError, match="boolean array"):
        Status.all_included(2).with_reason([True], Reason.NO_CONSENT)
    with pytest.raises(ValueError, match="boolean array"):
        Status.all_included(2).with_flag([0, 1], Flag.ALL_MIDPOINT)


def test_non_one_dimensional_included_raises():
    with pytest.raises(ValueError, match="one-dimensional"):
        Status(np.ones((1, 1), dtype=np.bool_), ((),), ((),))


@given(
    n=st.integers(0, 6),
    data=st.data(),
)
def test_status_invariants(n, data):
    s = Status.all_included(n)
    ops = data.draw(
        st.lists(
            st.tuples(
                st.booleans(),
                st.lists(st.booleans(), min_size=n, max_size=n),
                st.sampled_from(list(Reason)),
                st.sampled_from(list(Flag)),
            ),
            max_size=12,
        )
    )
    for is_reason, mask, reason, flag in ops:
        s = (
            s.with_reason(mask, reason) if is_reason else s.with_flag(mask, flag)
        )  # empty masks are lists of no booleans
    assert len(s.included) == len(s.reasons) == len(s.flags) == n
    assert s.included.tolist() == [len(r) == 0 for r in s.reasons]
    for row in s.reasons:
        assert list(row) == sorted(set(row), key=list(Reason).index)
    for row in s.flags:
        assert list(row) == sorted(set(row), key=list(Flag).index)
