"""Unit conversion for display: the real cases, every guard against a silently wrong number, rounding."""
import pytest

from summarizer import units


def metric(text):
    """The text for a metric / °C reader."""
    return units.convert(text, "metric", "c")


def imperial(text):
    """The text for an imperial / °F reader."""
    return units.convert(text, "imperial", "f")


@pytest.mark.parametrize("text,expected", [  # the summary that prompted this (890qpYAq1L8)
    ("one stored at 75° F", "one stored at 24 °C"),
    ("freezing until around -76° F, while", "freezing until around -60 °C, while"),
    ("a battery stored at 95 Fahrenheit discharges", "a battery stored at 35 °C discharges"),
    ("cranking amps at -22 Fahrenheit compared", "cranking amps at -30 °C compared"),
])
def test_the_real_cases(text, expected):
    assert metric(text) == expected
    assert imperial(text) == text  # already in the reader's units


@pytest.mark.parametrize("text,expected", [
    ("drove 120 miles", "drove 190 km"),  # "120" has two significant digits
    ("3.5 miles away", "5.6 km away"),
    ("0.4 miles", "640 m"),
    ("1,200 feet up", "370 m up"),
    ("2 feet of snow", "61 cm of snow"),
    ("0.5 inches of rain", "1.3 cm of rain"),
    ("at 75 mph", "at 120 km/h"),
    ("60 miles an hour", "97 km/h"),
    ("he weighs 150 lbs", "he weighs 68 kg"),
    ("lost 20 pounds of fat", "lost 9.1 kg of fat"),
    ("5 gallons of water", "19 L of water"),
    ("a cup of 8 fl oz", "a cup of 240 ml"),
    ("a 2,000 sq ft house", "a 190 m² house"),
    ("60-70 °F", "16–21 °C"),
    ("-22 to 75 Fahrenheit", "-30 to 24 °C"),
    ("dropped by 20 °F", "dropped by 11 °C"),
    ("he is 6 feet 2 inches tall", "he is 1.88 m tall"),
    ("she is 5'10\" tall", "she is 1.78 m tall"),
    ("a 6-foot-2 guard", "a 1.88 m guard"),
    ("75 °F (24 °C) today", "24 °C today"),
    ("it was −40 °F", "it was -40 °C"),
])
def test_imperial_to_metric(text, expected):
    assert metric(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("20 km away", "12 mi away"),
    ("about 6 metres tall", "about 20 ft tall"),
    ("he weighs 80 kg", "he weighs 180 lb"),
    ("90 km/h", "56 mph"),
    ("a 30 °C day", "a 86 °F day"),
    ("24 °C (75 °F)", "75 °F"),
])
def test_metric_to_imperial(text, expected):
    assert imperial(text) == expected


@pytest.mark.parametrize("text", [  # each would have become a wrong number
    'a quoted "$12,000" price and "$70, $80"',      # quote marks aren't inches
    "30 miles per gallon",                             # compound units
    "500 lb-ft of torque",
    "100 pounds per square inch",
    "a 1/4 inch gap and a ½ inch bolt",               # fractions
    "a 65-inch TV with 20-inch wheels on a 10-mile hike",  # nominal sizes
    "a 3.5 mm jack",
    "10m views and $5m raised",                        # bare symbols
    "a 12 oz can",                                     # ounces are ambiguous
    "earn 50,000 miles with the bonus",                # airline miles are points
    "it costs 500 pounds in London",                   # pounds as money
    "$5 per pound",
    "6 cubic feet",
    "six feet under, a few miles",                     # spelled numbers
    "75 degrees outside",                              # no scale
    "2 cups of flour and 3 tons",                      # left out on purpose
])
def test_traps_stay_as_written(text):
    assert metric(text) == text


def test_spoken_form_reads_units_as_words():
    assert metric("Hot: 95 °F and 20 km.") == "Hot: 35 °C and 20 km."
    spoken = units.convert("Hot: 95 °F and 20 km.", "metric", "c", spoken=True)
    assert spoken == "Hot: 35 degrees Celsius and 20 kilometres." and "°" not in spoken
    assert units.convert("60-70 °F", "metric", "c", spoken=True) == "16 to 21 degrees Celsius"


def test_mixed_settings():
    assert units.convert("20 km at 30 °C", "imperial", "c") == "12 mi at 30 °C"
    assert units.convert("20 miles at 86 °F", "metric", "f") == "32 km at 86 °F"


def test_nothing_to_convert():
    assert metric("") == "" and metric("No numbers here.") == "No numbers here."


def test_prompts_ask_for_measurements_as_in_the_source():
    from summarizer import summarize
    for prompt in (summarize.SYSTEM, summarize.BOOK_SYSTEM):
        assert "don't convert them" in prompt and "75 °F" in prompt


# ---------- in the bot ----------

async def test_the_same_summary_in_each_readers_units(app, telegram, monkeypatch):
    import asyncio
    import copy
    import access
    import bot
    from summarizer import db, pipeline, tts
    from conftest import msg_update, send
    for uid in (60, 61):
        access.set_state(uid, "allowed")
    db.set_user_units(61, "imperial", "f")
    monkeypatch.setattr(tts, "_ready", True)
    summary = {"title": "I Drove 1,000 Miles at 75° F", "is_clickbait": True,
               "clickbait_answer": "It ran at 95 Fahrenheit for 300 miles.",
               "summary": "• Stored at 75° F\n• Lost 20 pounds of weight", "_stats": {}}
    stored = copy.deepcopy(summary)
    monkeypatch.setattr(pipeline, "run", lambda url, *a, **k: pipeline.Result(
        "youtube", "abcdefghijk", url, {"title": "T"}, "", "captions", "en", summary))
    for uid in (60, 61):
        await send(app, msg_update(uid, "https://youtu.be/abcdefghijk"))
    task = asyncio.create_task(bot.worker(app))
    await asyncio.wait_for(bot.queue.join(), 10)
    task.cancel()
    ana, bob = ([m["text"] for m in telegram.sent("sendMessage") if m["chat_id"] == uid][-1] for uid in (60, 61))
    assert "35 °C for 480 km" in ana and "Stored at 24 °C" in ana and "9.1 kg" in ana
    assert "95 Fahrenheit for 300 miles" in bob and "75° F" in bob
    assert "I Drove 1,000 Miles at 75° F" in ana  # titles are names, not measurements
    assert summary == stored  # the cached summary itself never changes
    spoken = db.get_spoken(db.recent_requests(60)[0]["id"])["text"]
    assert "35 degrees Celsius" in spoken and "480 kilometres" in spoken


def test_chapters_are_converted_for_display():
    import bot
    from summarizer import documents
    r = documents.DocResult("each", "b.pdf", {"format": "pdf", "pages": 3},
                            chapters=[(0, "The 100-Mile Walk", "We walked 100 miles at 90 °F.")], llm="x")
    text = bot.render_document(r, units_=("metric", "c"))[0]
    assert "We walked 160 km at 32 °C." in text and "The 100-Mile Walk" in text


def test_defaults_and_migration():
    import sqlite3
    from summarizer import config, db
    assert db.get_user_units(12345) == ("metric", "c")  # unknown user: the defaults
    cols = {r[1] for r in sqlite3.connect(db.PATH).execute("PRAGMA table_info(users)")}
    assert {"unit_system", "temperature"} <= cols
    assert config.UNIT_SYSTEM == "metric" and config.TEMPERATURE == "c"


def test_bare_f_only_where_the_text_already_uses_fahrenheit():
    assert metric("survives -76° F, but freezes at 32F.") == "survives -60 °C, but freezes at 0 °C."
    assert metric("It got an F and a 5F rating.") == "It got an F and a 5F rating."  # no °F context
    assert metric("At 75 °F the F-150 starts.") == "At 24 °C the F-150 starts."


async def test_units_command_and_buttons(app, telegram):
    import access
    from summarizer import db
    from conftest import callback_update, msg_update, send
    access.set_state(60, "allowed")
    await send(app, msg_update(60, "/units"))
    first = telegram.sent("sendMessage")[-1]
    assert "Metric, temperatures in °C" in first["text"] and "✓ Metric" in str(first["reply_markup"])
    await send(app, callback_update(60, "units:sys:imperial"))
    await send(app, callback_update(60, "units:temp:f"))
    assert db.get_user_units(60) == ("imperial", "f")
    assert telegram.sent("answerCallbackQuery")[-1]["text"] == "Saved. Applies to new replies."
    assert "Imperial, temperatures in °F" in telegram.sent("editMessageText")[-1]["text"]
    await send(app, callback_update(60, "units:sys:kelvin"))  # forged
    await send(app, callback_update(99, "units:temp:c"))      # a stranger
    assert db.get_user_units(60) == ("imperial", "f")
    assert telegram.sent("answerCallbackQuery")[-1]["text"] == "This isn't available."
