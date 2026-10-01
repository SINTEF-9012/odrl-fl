"""A small SVG chart renderer, written here rather than imported.

The project has one plotting need -- five figures of two shapes -- and no
existing plotting dependency. matplotlib would add roughly 120 MB and a stack of
transitive packages to draw line charts with error bands, so this module draws
them directly instead.

That is a real trade and worth stating. What is given up: everything matplotlib
does that this does not, which is almost everything. What is bought is SVG that
is vector, tiny, deterministic, and reviewable as a text diff -- a figure whose
numbers changed shows up in `git diff` as changed numbers rather than as an
opaque binary blob. For a repository whose argument is that claims should be
checkable, that is the right side of the trade.

Colours are Okabe-Ito, which stays distinguishable under the common forms of
colour blindness. Series also differ in dash pattern, so the figures survive
being printed in greyscale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Okabe-Ito qualitative palette, ordered so the first few are maximally
#: distinct -- most figures here use three series or fewer.
PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#8C564B",  # brown
)

#: Dash patterns paired with the palette, so series stay separable in print.
DASHES = ("", "6 3", "2 3", "8 3 2 3", "12 4", "4 2", "1 3")

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"
FONT = "Helvetica, Arial, sans-serif"


def _esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@dataclass
class Series:
    """One line, optionally with a confidence band.

    Bands rather than error bars: with five series on one axis, bars overlap
    into noise, while bands stay readable and make it obvious where intervals
    overlap -- usually the question actually being asked of them.
    """

    label: str
    xs: list[float]
    ys: list[float]
    lo: list[float] | None = None
    hi: list[float] | None = None
    color: str | None = None
    dash: str | None = None
    markers: bool = True


@dataclass
class Bar:
    """One bar in a grouped bar chart."""

    group: str
    label: str
    value: float
    color: str | None = None
    annotation: str | None = None


@dataclass
class Axis:
    """Axis configuration. `ticks` overrides the automatic choice."""

    label: str = ""
    lo: float | None = None
    hi: float | None = None
    ticks: list[float] | None = None
    fmt: str = "{:g}"


def _floor_log10(value: float) -> int:
    if value <= 0:
        return 0
    exponent = 0
    while value < 1.0:
        value *= 10.0
        exponent -= 1
    while value >= 10.0:
        value /= 10.0
        exponent += 1
    return exponent


def nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Round tick positions covering [lo, hi].

    Steps are 1, 2, 2.5 or 5 times a power of ten -- the values a reader parses
    without effort. Anything else makes them do arithmetic to check a claim.
    """
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(target, 1)
    magnitude = 10.0 ** _floor_log10(raw)
    step = magnitude
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiple * magnitude
        if raw <= step:
            break
    start = step * int(lo / step)
    if start > lo:
        start -= step
    ticks = []
    value = start
    while value <= hi + step * 1e-9:
        if value >= lo - step * 1e-9:
            ticks.append(round(value, 10))
        value += step
    return ticks


#: Rough average glyph width at the note font size, in pixels. Measured rather
#: than guessed: the notes are the only wrapped text and they only need to not
#: overflow, so a conservative estimate beats a font-metrics dependency.
NOTE_CHAR_PX = 5.4


def wrap_note(text: str, width_px: float, char_px: float = NOTE_CHAR_PX) -> list[str]:
    """Break a note into lines that fit the plot width.

    SVG does not wrap text. A `<text>` element longer than the canvas is not
    clipped by the renderer so much as simply drawn past the edge, where the
    browser shows nothing and reports no error -- which is how five of these
    figures shipped with their notes cut off mid-sentence.
    """
    if not text:
        return []
    limit = max(int(width_px / char_px), 20)
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _bounds(values, lo, hi):
    if lo is None:
        lo = min(values)
    if hi is None:
        hi = max(values)
    if hi == lo:
        hi = lo + 1.0
    return lo, hi


def _header(width, height, title, subtitle, pad_l):
    opening = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="{FONT}">'
    )
    parts = [opening, f'<rect width="{width}" height="{height}" fill="#ffffff"/>']
    if title:
        parts.append(
            f'<text x="{pad_l}" y="24" font-size="15" font-weight="600" fill="{INK}">'
            f"{_esc(title)}</text>"
        )
    if subtitle:
        parts.append(
            f'<text x="{pad_l}" y="44" font-size="11.5" fill="{MUTED}">{_esc(subtitle)}</text>'
        )
    return parts


def line_chart(
    series: list[Series],
    *,
    title: str = "",
    subtitle: str = "",
    x: Axis | None = None,
    y: Axis | None = None,
    width: int = 720,
    height: int = 450,
    legend_title: str = "",
    notes: str = "",
) -> str:
    """Render line series with optional confidence bands."""
    x = x or Axis()
    y = y or Axis()

    pad_l, pad_r = 70, 20
    plot_w = width - pad_l - pad_r
    note_lines = wrap_note(notes, plot_w)
    pad_t = 36 + (16 if subtitle else 0)
    pad_b = 96 + 14 * len(note_lines)
    plot_h = height - pad_t - pad_b

    all_x = [v for s in series for v in s.xs]
    all_y = [v for s in series for v in s.ys]
    for s in series:
        all_y += list(s.lo or []) + list(s.hi or [])
    x_lo, x_hi = _bounds(all_x, x.lo, x.hi)
    y_lo, y_hi = _bounds(all_y, y.lo, y.hi)

    def px(value):
        return pad_l + (value - x_lo) / (x_hi - x_lo) * plot_w

    def py(value):
        return pad_t + plot_h - (value - y_lo) / (y_hi - y_lo) * plot_h

    parts = _header(width, height, title, subtitle, pad_l)

    for tick in (y.ticks if y.ticks is not None else nice_ticks(y_lo, y_hi)):
        yy = py(tick)
        parts.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + plot_w}" y2="{yy:.1f}" stroke="{GRID}" '
            'stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 9}" y="{yy + 3.5:.1f}" font-size="11" text-anchor="end" '
            f'fill="{MUTED}">{_esc(y.fmt.format(tick))}</text>'
        )

    for tick in (x.ticks if x.ticks is not None else nice_ticks(x_lo, x_hi)):
        parts.append(
            f'<text x="{px(tick):.1f}" y="{pad_t + plot_h + 18}" font-size="11" text-anchor="middle" '
            f'fill="{MUTED}">{_esc(x.fmt.format(tick))}</text>'
        )

    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="{INK}" '
        'stroke-width="1.2"/>'
    )

    if y.label:
        cy = pad_t + plot_h / 2
        parts.append(
            f'<text x="16" y="{cy:.1f}" font-size="11.5" fill="{INK}" '
            f'text-anchor="middle" transform="rotate(-90 16 {cy:.1f})">{_esc(y.label)}</text>'
        )
    if x.label:
        parts.append(
            f'<text x="{pad_l + plot_w / 2:.1f}" y="{pad_t + plot_h + 40}" font-size="11.5" text-anchor="middle" '
            f'fill="{INK}">{_esc(x.label)}</text>'
        )

    # Bands first, so the lines sit on top of them.
    for i, s in enumerate(series):
        color = s.color or PALETTE[i % len(PALETTE)]
        if s.lo and s.hi:
            top = " ".join(
                f"{px(a):.1f},{py(b):.1f}" for a, b in zip(s.xs, s.hi)
            )
            bottom = " ".join(
                f"{px(a):.1f},{py(b):.1f}"
                for a, b in reversed(list(zip(s.xs, s.lo)))
            )
            parts.append(
                f'<polygon points="{top} {bottom}" fill="{color}" fill-opacity="0.13" '
                'stroke="none"/>'
            )

    for i, s in enumerate(series):
        color = s.color or PALETTE[i % len(PALETTE)]
        dash = s.dash if s.dash is not None else DASHES[i % len(DASHES)]
        points = " ".join(
            f"{px(a):.1f},{py(b):.1f}" for a, b in zip(s.xs, s.ys)
        )
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash_attr} '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if s.markers:
            for a, b in zip(s.xs, s.ys):
                parts.append(
                    f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="2.6" fill="{color}"/>'
                )

    legend_y = pad_t + plot_h + 60
    cursor = pad_l
    if legend_title:
        parts.append(
            f'<text x="{cursor}" y="{legend_y + 4}" font-size="11" font-weight="600" '
            f'fill="{INK}">{_esc(legend_title)}</text>'
        )
        cursor += 7.0 * len(legend_title) + 16

    for i, s in enumerate(series):
        color = s.color or PALETTE[i % len(PALETTE)]
        dash = s.dash if s.dash is not None else DASHES[i % len(DASHES)]
        entry_w = 32 + 6.6 * len(s.label)
        if cursor + entry_w > width - pad_r:
            cursor = pad_l
            legend_y += 19
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{cursor}" y1="{legend_y}" x2="{cursor + 20}" y2="{legend_y}" stroke="{color}" '
            f'stroke-width="2.4"{dash_attr}/>'
        )
        parts.append(
            f'<text x="{cursor + 26}" y="{legend_y + 4}" font-size="11" fill="{INK}">{_esc(s.label)}</text>'
        )
        cursor += entry_w

    for i, line in enumerate(note_lines):
        y = height - 10 - 14 * (len(note_lines) - 1 - i)
        parts.append(
            f'<text x="{pad_l}" y="{y}" font-size="10.5" '
            f'fill="{MUTED}">{_esc(line)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def grouped_bar_chart(
    bars: list[Bar],
    *,
    title: str = "",
    subtitle: str = "",
    y: Axis | None = None,
    width: int = 720,
    height: int = 430,
    notes: str = "",
    reference: tuple[float, str] | None = None,
    sublabels: dict[str, str] | None = None,
) -> str:
    """Render bars grouped along the x axis.

    `sublabels` puts a second line under a group name. Facts about a *group*
    ("blind selection was usable 20% of the time") go there rather than onto a
    bar: a zero-height bar has nowhere to put a label, and long text on adjacent
    bars collides.

    `reference` draws a labelled horizontal rule -- used for the noise floor on
    the TCGA figure, where the whole point is that the bars differ by less than
    the measurement error.
    """
    y = y or Axis()
    sublabels = sublabels or {}
    pad_l, pad_r = 70, 20
    plot_w = width - pad_l - pad_r
    note_lines = wrap_note(notes, plot_w)
    pad_t = 36 + (16 if subtitle else 0)
    pad_b = 88 + 14 * len(note_lines) + (14 if sublabels else 0)
    plot_h = height - pad_t - pad_b

    groups: list[str] = []
    labels: list[str] = []
    for bar in bars:
        if bar.group not in groups:
            groups.append(bar.group)
        if bar.label not in labels:
            labels.append(bar.label)

    values = [b.value for b in bars] + ([reference[0]] if reference else [])
    y_lo, y_hi = _bounds(values, 0.0 if y.lo is None else y.lo, y.hi)
    # Headroom for the value labels, which sit above each bar. Without it the
    # tallest bar's label collides with the subtitle.
    if y.hi is None:
        y_hi += (y_hi - y_lo) * 0.12

    def py(value):
        return pad_t + plot_h - (value - y_lo) / (y_hi - y_lo) * plot_h

    parts = _header(width, height, title, subtitle, pad_l)

    for tick in (y.ticks if y.ticks is not None else nice_ticks(y_lo, y_hi)):
        yy = py(tick)
        parts.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + plot_w}" y2="{yy:.1f}" stroke="{GRID}" '
            'stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 9}" y="{yy + 3.5:.1f}" font-size="11" text-anchor="end" '
            f'fill="{MUTED}">{_esc(y.fmt.format(tick))}</text>'
        )

    group_w = plot_w / max(len(groups), 1)
    bar_w = min(64.0, group_w / (len(labels) + 0.6))
    for bar in bars:
        gi = groups.index(bar.group)
        li = labels.index(bar.label)
        span = bar_w * len(labels)
        x0 = pad_l + gi * group_w + (group_w - span) / 2 + li * bar_w
        top = py(bar.value)
        color = bar.color or PALETTE[li % len(PALETTE)]
        # A bar of exactly zero still has to be visible, or the reader cannot
        # tell it apart from a bar that was never drawn. Two pixels says
        # "measured, and it is nothing".
        drawn_h = max(pad_t + plot_h - top, 2.0)
        parts.append(
            f'<rect class="bar" x="{x0:.1f}" y="{pad_t + plot_h - drawn_h:.1f}" '
            f'width="{bar_w - 6:.1f}" height="{drawn_h:.1f}" fill="{color}" rx="2"/>'
        )
        text = bar.annotation if bar.annotation is not None else y.fmt.format(bar.value)
        parts.append(
            f'<text x="{x0 + (bar_w - 6) / 2:.1f}" y="{pad_t + plot_h - drawn_h - 6:.1f}" '
            f'font-size="10.5" font-weight="600" text-anchor="middle" '
            f'fill="{INK}">{_esc(text)}</text>'
        )

    for gi, group in enumerate(groups):
        cx = pad_l + gi * group_w + group_w / 2
        parts.append(
            f'<text x="{cx:.1f}" y="{pad_t + plot_h + 20}" font-size="11.5" '
            f'text-anchor="middle" fill="{INK}">{_esc(group)}</text>'
        )
        if sublabels and group in sublabels:
            parts.append(
                f'<text x="{cx:.1f}" y="{pad_t + plot_h + 35}" font-size="10.5" '
                f'text-anchor="middle" fill="{MUTED}">{_esc(sublabels[group])}</text>'
            )

    if reference:
        value, label = reference
        yy = py(value)
        parts.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l + plot_w}" y2="{yy:.1f}" stroke="{INK}" '
            'stroke-width="1.3" stroke-dasharray="5 4"/>'
        )
        parts.append(
            f'<text x="{pad_l + plot_w - 4}" y="{yy - 6:.1f}" font-size="10.5" text-anchor="end" '
            f'fill="{INK}">{_esc(label)}</text>'
        )

    parts.append(
        f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="{INK}" '
        'stroke-width="1.2"/>'
    )
    if y.label:
        cy = pad_t + plot_h / 2
        parts.append(
            f'<text x="16" y="{cy:.1f}" font-size="11.5" fill="{INK}" '
            f'text-anchor="middle" transform="rotate(-90 16 {cy:.1f})">{_esc(y.label)}</text>'
        )

    legend_y = pad_t + plot_h + 48 + (14 if sublabels else 0)
    cursor = pad_l
    for li, label in enumerate(labels):
        color = next(
            (b.color for b in bars if b.label == label and b.color),
            PALETTE[li % len(PALETTE)],
        )
        parts.append(
            f'<rect x="{cursor}" y="{legend_y - 8}" width="12" height="12" fill="{color}" rx="2"/>'
        )
        parts.append(
            f'<text x="{cursor + 18}" y="{legend_y + 2}" font-size="11" fill="{INK}">{_esc(label)}</text>'
        )
        cursor += 32 + 6.6 * len(label)

    for i, line in enumerate(note_lines):
        y = height - 10 - 14 * (len(note_lines) - 1 - i)
        parts.append(
            f'<text x="{pad_l}" y="{y}" font-size="10.5" '
            f'fill="{MUTED}">{_esc(line)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def save(svg: str, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path
