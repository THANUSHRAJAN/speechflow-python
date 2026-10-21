from app.transcriber import dedupe_overlap, join_text


def test_dedupe_removes_repeated_tail_words():
    existing = "we are going to the market today"
    new = "to the market today and then home"
    assert dedupe_overlap(existing, new, min_words=3) == "and then home"


def test_dedupe_ignores_case_and_punctuation():
    assert dedupe_overlap("see you Tomorrow, okay", "tomorrow okay, bye", min_words=2) == "bye"


def test_dedupe_keeps_text_without_overlap():
    assert dedupe_overlap("hello there", "completely different words", min_words=1) == "completely different words"


def test_dedupe_respects_min_words():
    # a single repeated word is legitimate repetition unless the caller allows it (forced commit)
    assert dedupe_overlap("very good", "good idea", min_words=3) == "good idea"
    assert dedupe_overlap("very good", "good idea", min_words=1) == "idea"


def test_dedupe_handles_hindi_text():
    existing = "मैं आज बाजार जा रहा हूँ"
    new = "बाजार जा रहा हूँ और फिर घर"
    assert dedupe_overlap(existing, new, min_words=3) == "और फिर घर"


def test_dedupe_char_level_for_spaceless_languages():
    assert dedupe_overlap("今日は天気がとても良い", "天気がとても良いですね", char_level=True) == "ですね"


def test_dedupe_empty_inputs():
    assert dedupe_overlap("", "abc") == "abc"
    assert dedupe_overlap("abc", "") == ""


def test_join_text_uses_language_specific_separator():
    assert join_text("hello", "world", "en") == "hello world"
    assert join_text("今日は", "晴れ", "ja") == "今日は晴れ"
    assert join_text("", "first", "en") == "first"
    assert join_text("only", "", "en") == "only"
