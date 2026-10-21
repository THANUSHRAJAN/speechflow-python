import asyncio

from app.config import SAMPLE_RATE
from app.transcriber import StreamingSession
from tests.conftest import FakeEngine, result, tone


def run(coro):
    return asyncio.run(coro)


def test_stable_segments_are_committed_and_rest_stays_interim(settings):
    engine = FakeEngine([result((0.0, 3.0, "hello world how are you"), (3.5, 9.5, "this is a test"))])
    session = StreamingSession(engine, settings, "en")
    session.add_audio(tone(10))

    events = run(session.update())

    final = [e for e in events if e["type"] == "final"][0]
    partial = [e for e in events if e["type"] == "partial"][0]
    assert final["text"] == "hello world how are you"
    assert final["fullText"] == "hello world how are you"
    assert partial["text"] == "this is a test"
    # the 3 committed seconds are removed from the buffer
    assert abs(session.buffer_seconds - 7.0) < 0.01


def test_final_flush_commits_everything_and_clears_buffer(settings):
    engine = FakeEngine([
        result((0.0, 3.0, "first sentence here"), (3.5, 9.5, "second sentence")),
        result((0.0, 6.0, "second sentence")),
    ])
    session = StreamingSession(engine, settings, "en")
    session.add_audio(tone(10))
    run(session.update())
    events = run(session.update(final=True))

    assert [e["type"] for e in events] == ["final"]
    assert session.full_text == "first sentence here second sentence"
    assert session.buffer_seconds == 0


def test_forced_commit_keeps_overlap_and_dedupes_repeated_words(settings):
    engine = FakeEngine([
        result((0.0, 8.0, "one long continuous run of words without any pause")),
        result((0.0, 4.0, "without any pause and then it continues")),
    ])
    session = StreamingSession(engine, settings, "en")
    session.add_audio(tone(8))  # == max_buffer, one segment, nothing "stable" -> forced commit
    first = run(session.update())

    assert session.full_text == "one long continuous run of words without any pause"
    assert abs(session.buffer_seconds - 1.0) < 0.01  # overlap kept
    assert [e["type"] for e in first] == ["final", "partial"]

    session.add_audio(tone(3))
    run(session.update(final=True))
    assert session.full_text == "one long continuous run of words without any pause and then it continues"


def test_silence_never_reaches_the_model(settings):
    engine = FakeEngine()
    session = StreamingSession(engine, settings, "en")
    session.add_audio(b"\x00\x00" * SAMPLE_RATE * 5)

    assert run(session.update()) == []
    assert engine.calls == []
    assert session.buffer_seconds <= 1.01  # long silence is trimmed so the buffer cannot grow forever


def test_update_scheduling_thresholds(settings):
    session = StreamingSession(FakeEngine(), settings, "en")
    assert not session.ready_for_update()
    session.add_audio(tone(0.5))
    assert not session.ready_for_update()
    session.add_audio(tone(0.6))
    assert session.ready_for_update()


def test_auto_language_locks_once_confident(settings):
    engine = FakeEngine([
        result((0.0, 2.0, "नमस्ते दोस्तों"), language="hi", prob=0.93),
        result((0.0, 3.0, "आप कैसे हैं"), language="hi", prob=0.95),
    ])
    session = StreamingSession(engine, settings, "auto")
    session.add_audio(tone(3))
    run(session.update())
    session.add_audio(tone(1))
    run(session.update())

    assert engine.calls[0][1] is None  # first pass: auto-detect
    assert engine.calls[1][1] == "hi"  # then locked
    assert session.language == "hi"


def test_prompt_contains_previous_committed_text(settings):
    engine = FakeEngine([result((0.0, 2.0, "committed words here")), result()])
    session = StreamingSession(engine, settings, "en")
    session.add_audio(tone(5))
    run(session.update())
    session.add_audio(tone(2))
    run(session.update())

    assert engine.calls[1][2] == "committed words here"


def test_odd_byte_count_is_tolerated(settings):
    session = StreamingSession(FakeEngine(), settings, "en")
    session.add_audio(b"\x01\x02\x03")
    assert session.buffer_seconds == 1 / SAMPLE_RATE
