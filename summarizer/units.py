"""Shows measurements in each user's units: metric or imperial, °C or °F (see /units).

Summaries are cached once per video and model and shared, so they keep the units the video used; the text
is converted for each reader when it's shown, by rules, without the AI. Only the converted value is shown,
so a wrong conversion would be a silently wrong number: only clear cases are converted (a number in digits
directly followed by an unambiguous unit), and every known trap has a guard. When in doubt the text stays as
written. Conversion is display-only: the stored summary never changes.
"""
import math
import re
from dataclasses import dataclass

# ---------- units ----------


@dataclass(frozen=True)
class Unit:
    """One unit the converter recognises.

    Attributes:
        kind: temperature, length, distance, speed, weight, volume, small_volume, area.
        system: "metric" or "imperial".
        factor: Value in the kind's metric base unit per 1 of this unit (temperature: unused).
        symbol: How a value in this unit is written in output.
        words: How it's read aloud (🔊).
    """
    kind: str
    system: str
    factor: float
    symbol: str
    words: str


UNITS = {
    "f": Unit("temperature", "imperial", 1, "°F", "degrees Fahrenheit"),
    "c": Unit("temperature", "metric", 1, "°C", "degrees Celsius"),
    "mph": Unit("speed", "imperial", 1.609344, "mph", "miles per hour"),
    "kmh": Unit("speed", "metric", 1, "km/h", "kilometres per hour"),
    "sqft": Unit("area", "imperial", 0.09290304, "sq ft", "square feet"),
    "m2": Unit("area", "metric", 1, "m²", "square metres"),
    "floz": Unit("small_volume", "imperial", 29.5735295625, "fl oz", "fluid ounces"),
    "ml": Unit("small_volume", "metric", 1, "ml", "millilitres"),
    "gal": Unit("volume", "imperial", 3.785411784, "gal", "gallons"),
    "l": Unit("volume", "metric", 1, "L", "litres"),
    "mi": Unit("distance", "imperial", 1.609344, "mi", "miles"),
    "km": Unit("distance", "metric", 1, "km", "kilometres"),
    "ft": Unit("length", "imperial", 0.3048, "ft", "feet"),
    "inch": Unit("length", "imperial", 0.0254, "in", "inches"),
    "m": Unit("length", "metric", 1, "m", "metres"),
    "cm": Unit("length", "metric", 0.01, "cm", "centimetres"),
    "mm": Unit("length", "metric", 0.001, "mm", "millimetres"),
    "lb": Unit("weight", "imperial", 0.45359237, "lb", "pounds"),
    "kg": Unit("weight", "metric", 1, "kg", "kilograms"),
}
# Written forms, longest first within a kind (so "miles per hour" wins over "miles"). Bare symbols that mean
# other things are deliberately absent: m (10m views), g, L, in (a preposition), k, M, oz (fluid/weight/troy).
_FORMS = [
    ("f", r"°\s?F(?![A-Za-z])|°?\s?(?:degrees?\s+)?Fahrenheit|degrees?\s+F(?![A-Za-z])"),
    ("c", r"°\s?C(?![A-Za-z])|°?\s?(?:degrees?\s+)?(?:Celsius|centigrade)|degrees?\s+C(?![A-Za-z])"),
    ("mph", r"mph|miles\s+(?:per|an)\s+hour"),
    ("kmh", r"km/h|kph|km\s+per\s+hour|kilomet(?:er|re)s\s+(?:per|an)\s+hour"),
    ("sqft", r"sq\.?\s?ft|square\s+f(?:ee|oo)t|ft²"),
    ("m2", r"m²|sq\.?\s?m(?![A-Za-z])|square\s+met(?:er|re)s?"),
    ("floz", r"fl\.?\s?oz|fluid\s+ounces?"),
    ("ml", r"ml|millilit(?:er|re)s?"),
    ("gal", r"gallons?|gal(?![A-Za-z])"),
    ("l", r"lit(?:er|re)s?"),
    ("mi", r"miles?|mi(?![A-Za-z])"),
    ("km", r"km|kilomet(?:er|re)s?"),
    ("ft", r"feet|foot|ft(?![A-Za-z])"),
    ("inch", r"inch(?:es)?"),
    ("m", r"met(?:er|re)s"),
    ("cm", r"cm|centimet(?:er|re)s?"),
    ("mm", r"mm|millimet(?:er|re)s?"),
    ("lb", r"lbs?|pounds?"),
    ("kg", r"kg|kilo(?:gram)?s?"),
]
_UNIT_ALT = "|".join(f"(?P<u_{key}>{form})" for key, form in _FORMS)

# A number in digits: sign (also U+2212), "1,200" (groups of exactly three), decimals. Not inside another token
# ("v2", "1/4", "x.5"), and not money ("$5 per pound").
_NUM_BODY = r"[-−]?(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?"
_NUM = r"(?<![\w/.,½¼¾$£€])" + _NUM_BODY
_QTY = re.compile(
    rf"(?P<n1>{_NUM})(?:\s*(?:-|–|−|to)\s*(?P<n2>{_NUM_BODY}))?"
    rf"\s?(?:{_UNIT_ALT})(?![A-Za-z])"  # a whole unit: never "mile" out of "miles"
    # Part of a compound unit (miles per gallon, lb-ft, pounds per square inch): not ours, left as written.
    r"(?![-/][A-Za-z])(?!\s+per\s)(?!\s+an?\s+gallon)"
    r"(?P<paren>\s*\((?P<inner>[^()]{1,30})\))?",
    re.I)
# A height: 6 feet 2 inches, 6 ft 2 in, 6'2", 6-foot-2.
_HEIGHT = re.compile(r"(?<![\w.])(?P<ft>[4-8])(?:\s*(?:feet|foot|ft|['′])\s*|-foot-)(?P<in>1[01]|\d)"
                     r"(?:\s*(?:inches|inch|in(?![A-Za-z])|[\"″]))?(?![\d.])", re.I)
_EXPLICIT_F = re.compile(r"\d\s?°\s?F(?![A-Za-z])|\d\s?(?:degrees\s+)?Fahrenheit", re.I)
_BARE_F = re.compile(r"(?<![\w.,/$£€])(-?\d+(?:\.\d+)?)F(?![\w-])")
_DELTA_BEFORE = re.compile(r"\b(?:by|difference of|swing of|swings? by|rise of|drop of|change of)\s*$", re.I)
_DELTA_AFTER = re.compile(r"^\s*(?:warmer|cooler|hotter|colder|higher|lower|difference|swing|rise|drop)\b", re.I)
_WEIGHT_CUE = re.compile(r"weigh|weight|\blost\b|\bgained?\b|\blift|heavy|squat|bench|press|deadlift|\bbody\b|"
                         r"\bfat\b|muscle|\bbaby\b|\bmeat\b|flour|\bbag\b|\bload\b|payload|\btow", re.I)
_POINTS_CUE = re.compile(r"\bpoints?\b|\baward\b|redeem|\bbonus\b|frequent[- ]fl[yi]er|airline miles", re.I)


def _sentence(text: str, start: int, end: int) -> str:
    """The sentence (or line) around a match, for the context guards."""
    left = max(text.rfind(c, 0, start) for c in ".!?\n")
    rights = [i for i in (text.find(c, end) for c in ".!?\n") if i != -1]
    return text[left + 1:min(rights) if rights else len(text)]


# ---------- numbers ----------

def _value(s: str) -> float:
    """A matched number as a float."""
    return float(s.replace(",", "").replace("−", "-"))


def _significant(s: str) -> int:
    """Significant digits of a written number, at least 2 (trailing zeros of an integer don't count)."""
    digits = s.replace(",", "").lstrip("-−")
    if "." in digits:
        count = len(digits.replace(".", "").lstrip("0"))
    else:
        count = len(digits.lstrip("0").rstrip("0"))
    return max(2, count)


def _fmt(x: float, sig: int) -> str:
    """A value rounded to `sig` significant digits, with thousands separators and no trailing zeros."""
    if x == 0:
        return "0"
    places = sig - 1 - math.floor(math.log10(abs(x)))  # negative: round to tens, hundreds…
    r = round(x, places)
    text = f"{r:,.{max(0, places)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


# ---------- conversion ----------

def _target(unit: Unit, system: str, temperature: str) -> str:
    """The system a value of this unit should be shown in."""
    if unit.kind == "temperature":
        return "imperial" if temperature == "f" else "metric"
    return system


def _out_unit(src: str, x: float) -> str:
    """The unit to show a converted value in, by the source unit and the size of the value."""
    if src == "mi":
        return "km" if abs(x) * UNITS["mi"].factor >= 1 else "m"
    if src in ("ft", "inch"):
        metres = abs(x) * UNITS[src].factor
        return "mm" if metres < 0.01 else "cm" if metres < 1 else "m"
    return {"lb": "kg", "gal": "l", "floz": "ml", "km": "mi", "m": "ft", "cm": "inch", "mm": "inch", "kg": "lb",
            "l": "gal", "ml": "floz", "kmh": "mph", "mph": "kmh", "sqft": "m2", "m2": "sqft"}[src]


def _convert_value(x: float, src: str, dst: str, delta: bool) -> float:
    """Converts a value between two units of the same kind."""
    if src == "f":
        return x * 5 / 9 if delta else (x - 32) * 5 / 9
    if src == "c":
        return x * 9 / 5 if delta else x * 9 / 5 + 32
    s, d = UNITS[src], UNITS[dst]
    if s.kind == "distance" and d.kind == "length":  # miles -> metres when under a km
        return x * s.factor * 1000
    if s.kind == "length" and d.kind == "distance":
        return x * s.factor / 1000
    return x * s.factor / d.factor


def _render(values: list[str], unit_key: str, spoken: bool) -> str:
    """Formats one value or a range in a unit."""
    unit = UNITS[unit_key]
    if spoken:  # "-30" may be read as "dash thirty", or the sign dropped
        values = [f"minus {v[1:]}" if v.startswith("-") else v for v in values]
    # "16–21 °C", but "-30 to 24 °C": a dash next to a minus sign reads as a negative number.
    joined = " to ".join(values) if spoken or any(v.startswith("-") for v in values) else "–".join(values)
    return f"{joined} {unit.words}" if spoken else f"{joined} {unit.symbol}"


def _paren_is_other_system(inner: str, unit: Unit) -> bool:
    """Whether a parenthesis after a quantity holds it in the other system (a conversion the text gave)."""
    m = _QTY.search(inner)
    if not m:
        return False
    other = UNITS[next(k[2:] for k, v in m.groupdict().items() if k.startswith("u_") and v)]
    same_kind = other.kind == unit.kind or {other.kind, unit.kind} <= {"length", "distance"}
    return same_kind and other.system != unit.system


def convert(text: str, system: str = "metric", temperature: str = "c", spoken: bool = False) -> str:
    """Rewrites the measurements in a text into the reader's units.

    Args:
        text: Model-written text (a summary, a clickbait answer, a chapter summary). Not titles: those are
            names ("500 Miles").
        system: "metric" or "imperial" for everything but temperatures.
        temperature: "c" or "f".
        spoken: Units as words ("24 degrees Celsius"), for reading aloud; also rewrites values already in
            the reader's units, whose symbols would be read badly.

    Returns:
        The text with converted values (only the converted value is shown).
    """
    if not text:
        return text
    # "32F" (no degree sign) is a temperature only when the text already gives temperatures in °F: a bare F
    # is otherwise a grade, a size or part of a name, so it's left alone.
    if _EXPLICIT_F.search(text):
        text = _BARE_F.sub(lambda m: f"{m[1]} °F", text)
    if system == "metric":
        def height(m: re.Match) -> str:
            """A feet-and-inches height as one metric value."""
            metres = (int(m["ft"]) * 12 + int(m["in"])) * 0.0254
            return f"{metres:.2f} metres" if spoken else f"{metres:.2f} m"

        text = _HEIGHT.sub(height, text)

    def replace(m: re.Match) -> str:
        """One quantity, converted, kept, or (spoken) written out."""
        key = next(k[2:] for k, v in m.groupdict().items() if k.startswith("u_") and v)
        unit = UNITS[key]
        whole = m.group(0)
        paren = m["paren"] or ""
        core = whole[:len(whole) - len(paren)]
        drop_paren = bool(paren) and _paren_is_other_system(m["inner"], unit)
        tail = "" if drop_paren else paren
        sentence = _sentence(m.string, m.start(), m.end())
        unit_text = m[f"u_{key}"].lower()
        # Words that are often not measurements in a given sentence: leave those as written.
        if key == "lb" and unit_text.startswith("pound") and not _WEIGHT_CUE.search(sentence):
            return whole  # "500 pounds" is a UK price as often as a weight
        if key == "mi" and _POINTS_CUE.search(sentence):
            return whole  # airline miles are points
        numbers = [n for n in (m["n1"], m["n2"]) if n]
        if _target(unit, system, temperature) == unit.system:
            if not spoken:
                return core + tail
            return _render([n.replace("−", "-") for n in numbers], key, True) + tail
        if key == "mm" and abs(_value(numbers[-1])) < 10:
            return whole  # 3.5 mm jacks, 9 mm calibres: names of sizes
        before = m.string[max(0, m.start() - 20):m.start()]
        after = m.string[m.end():m.end() + 20]
        delta = unit.kind == "temperature" and bool(_DELTA_BEFORE.search(before) or _DELTA_AFTER.search(after))
        if unit.kind == "temperature":
            dst = "f" if key == "c" else "c"
            out = [str(round(_convert_value(_value(n), key, dst, delta)) + 0) for n in numbers]
            return _render(out, dst, spoken) + tail
        dst = _out_unit(key, max(abs(_value(n)) for n in numbers))  # a range shares the larger end's unit
        out = [_fmt(_convert_value(_value(n), key, dst, False), _significant(n)) for n in numbers]
        return _render(out, dst, spoken) + tail

    return _QTY.sub(replace, text)
