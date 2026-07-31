"""ClauseAggregator checks — runnable without the model stack.

    PYTHONPATH=. python tests/test_clause.py     # or: pytest tests/
"""

from __future__ import annotations

from voice_agent.llm.clause import ClauseAggregator


def test_emits_on_sentence_boundary() -> None:
    agg = ClauseAggregator()
    assert agg.push("Hello there") == []          # no boundary yet
    out = agg.push(", how can I help you today? ")
    assert out == ["Hello there, how can I help you today?"]
    assert agg.flush() == ""


def test_streamed_token_by_token() -> None:
    agg = ClauseAggregator()
    emitted: list[str] = []
    for tok in ["The ", "order ", "ships ", "today. ", "Anything ", "else?"]:
        emitted += agg.push(tok)
    emitted.append(agg.flush())
    emitted = [e for e in emitted if e]
    assert emitted == ["The order ships today.", "Anything else?"]


def test_decimal_point_is_not_a_boundary() -> None:
    agg = ClauseAggregator()
    # "19.90" must not split; the trailing '.' after "euros" ends the clause.
    out = agg.push("It costs 19.90 euros today. ")
    assert out == ["It costs 19.90 euros today."]


def test_greek_question_mark_boundary() -> None:
    agg = ClauseAggregator()
    out = agg.push("Πώς μπορώ να βοηθήσω σήμερα; ")
    assert out == ["Πώς μπορώ να βοηθήσω σήμερα;"]


def test_soft_cap_without_punctuation() -> None:
    agg = ClauseAggregator(min_chars=12, max_chars=40)
    long_run = "word " * 20  # 100 chars, no sentence punctuation
    out = agg.push(long_run)
    assert out, "expected the soft cap to emit at least one chunk"
    assert all(len(c) <= 40 for c in out)


def test_short_fragment_waits_for_flush() -> None:
    agg = ClauseAggregator(min_chars=12)
    assert agg.push("Ναι.") == []        # below min_chars, hold it
    assert agg.flush() == "Ναι."


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
