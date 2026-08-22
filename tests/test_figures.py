"""Tests for the SVG chart renderer.

A chart library that silently draws the wrong thing is worse than no chart
library, and "it produced a file" is not evidence it drew anything correct. So
these check the parts that can be verified without eyes: that output is
well-formed XML, that data maps to coordinates the way the axes claim, and that
the specific rendering bugs already found once do not come back.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from itertools import pairwise

import pytest

from odrl_fl.figures import (
    PALETTE,
    Axis,
    Bar,
    Series,
    grouped_bar_chart,
    line_chart,
    nice_ticks,
    save,
)


def parse(svg: str) -> ET.Element:
    """Parsing is the cheapest real check: malformed SVG renders as nothing."""
    return ET.fromstring(svg)


def bars_of(root: ET.Element) -> list[ET.Element]:
    """Only the bars. Legend swatches are rects too, and an earlier version of
    these tests matched those instead -- reporting a 12x12 swatch as the bar."""
    return [e for e in root.iter() if e.get("class") == "bar"]


class TestNiceTicks:
    def test_unit_interval_gets_readable_steps(self):
        assert nice_ticks(0, 1) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def test_ticks_span_the_requested_range(self):
        ticks = nice_ticks(0, 30)
        assert ticks[0] <= 0 and ticks[-1] >= 30

    def test_steps_are_uniform(self):
        ticks = nice_ticks(0, 30)
        steps = {round(b - a, 9) for a, b in pairwise(ticks)}
        assert len(steps) == 1

    def test_degenerate_range_does_not_hang(self):
        """A flat series is real data, not a bug -- h = 0 produces one."""
        assert nice_ticks(5, 5) == [5]

    @pytest.mark.parametrize("hi", [0.001, 1, 7, 250, 9999])
    def test_steps_stay_round_across_magnitudes(self, hi):
        ticks = nice_ticks(0, hi)
        step = ticks[1] - ticks[0]
        mantissa = step / 10 ** (len(str(int(step))) - 1) if step >= 1 else None
        assert len(ticks) >= 2
        if mantissa is not None:
            assert round(mantissa, 6) in (1.0, 2.0, 2.5, 5.0)


class TestLineChart:
    def test_output_is_well_formed(self):
        svg = line_chart([Series("a", [1, 2, 3], [0.1, 0.5, 0.9])])
        assert parse(svg).tag.endswith("svg")

    def test_a_series_becomes_a_polyline_with_every_point(self):
        svg = line_chart([Series("a", [1, 2, 3], [0.1, 0.5, 0.9])])
        polylines = [e for e in parse(svg).iter() if e.tag.endswith("polyline")]
        assert len(polylines) == 1
        assert len(polylines[0].get("points").split()) == 3

    def test_y_axis_is_inverted_because_svg_grows_downward(self):
        """The bug that makes every chart silently upside down."""
        svg = line_chart([Series("a", [1, 2], [0.0, 1.0])], y=Axis(lo=0, hi=1))
        points = next(
            e for e in parse(svg).iter() if e.tag.endswith("polyline")
        ).get("points")
        first, second = (p.split(",") for p in points.split())
        assert float(second[1]) < float(first[1])

    def test_a_band_is_drawn_when_bounds_are_given(self):
        svg = line_chart(
            [Series("a", [1, 2], [0.5, 0.5], lo=[0.4, 0.4], hi=[0.6, 0.6])]
        )
        polygons = [e for e in parse(svg).iter() if e.tag.endswith("polygon")]
        assert len(polygons) == 1
        # Closes the loop: forward along hi, back along lo.
        assert len(polygons[0].get("points").split()) == 4

    def test_no_band_without_bounds(self):
        svg = line_chart([Series("a", [1, 2], [0.5, 0.5])])
        assert not [e for e in parse(svg).iter() if e.tag.endswith("polygon")]

    def test_series_get_distinct_colours(self):
        svg = line_chart([Series(str(i), [1, 2], [0, 1]) for i in range(4)])
        strokes = {
            e.get("stroke")
            for e in parse(svg).iter()
            if e.tag.endswith("polyline")
        }
        assert len(strokes) == 4
        assert strokes <= set(PALETTE)

    def test_series_also_differ_by_dash_for_greyscale_printing(self):
        svg = line_chart([Series(str(i), [1, 2], [0, 1]) for i in range(4)])
        dashes = {
            e.get("stroke-dasharray")
            for e in parse(svg).iter()
            if e.tag.endswith("polyline")
        }
        assert len(dashes) == 4

    def test_labels_are_escaped(self):
        """Labels come from CSV columns, so they are not trusted markup."""
        svg = line_chart([Series("a<b>&c", [1, 2], [0, 1])])
        assert "a<b>" not in svg
        assert "&amp;c" in svg
        parse(svg)

    def test_percentage_formatting_reaches_the_axis(self):
        svg = line_chart(
            [Series("a", [1, 2], [0.0, 1.0])],
            y=Axis(lo=0, hi=1, fmt="{:.0%}"),
        )
        assert "100%" in svg


class TestGroupedBarChart:
    def test_output_is_well_formed(self):
        svg = grouped_bar_chart([Bar("g", "a", 1.0)])
        assert parse(svg).tag.endswith("svg")

    def test_bar_height_is_proportional_to_value(self):
        svg = grouped_bar_chart(
            [Bar("g", "a", 1.0), Bar("g", "b", 2.0)], y=Axis(lo=0, hi=2)
        )
        rects = bars_of(parse(svg))
        heights = sorted(float(r.get("height")) for r in rects)
        assert heights[1] == pytest.approx(2 * heights[0], rel=0.02)

    def test_a_zero_bar_is_still_visible(self):
        """A zero-height rect is indistinguishable from a missing one, and
        'deployable utility is zero' is a result, not an absence."""
        svg = grouped_bar_chart([Bar("g", "a", 0.0), Bar("g", "b", 10.0)])
        rects = bars_of(parse(svg))
        assert min(float(r.get("height")) for r in rects) >= 2.0

    def test_tall_bars_leave_room_for_their_labels(self):
        """The label sits above the bar, so the tallest bar must not reach the
        top of the plot or the label lands on the subtitle."""
        svg = grouped_bar_chart([Bar("g", "a", 100.0)], subtitle="sub")
        rect = bars_of(parse(svg))[0]
        assert float(rect.get("y")) > 60

    def test_an_explicit_upper_bound_is_respected(self):
        """Headroom is automatic only when the caller did not pin the axis."""
        svg = grouped_bar_chart([Bar("g", "a", 0.8)], y=Axis(lo=0, hi=0.8))
        assert "0.8" in svg

    def test_sublabels_appear_under_their_group(self):
        svg = grouped_bar_chart(
            [Bar("mygroup", "a", 1.0)], sublabels={"mygroup": "usable 20%"}
        )
        assert "usable 20%" in svg
        parse(svg)

    def test_a_reference_rule_is_drawn_and_labelled(self):
        svg = grouped_bar_chart(
            [Bar("g", "a", 0.7)], reference=(0.5, "chance"), y=Axis(lo=0, hi=1)
        )
        dashed = [
            e
            for e in parse(svg).iter()
            if e.tag.endswith("line") and e.get("stroke-dasharray")
        ]
        assert len(dashed) == 1
        assert "chance" in svg


class TestSave:
    def test_save_writes_and_creates_parents(self, tmp_path):
        target = tmp_path / "nested" / "fig.svg"
        save(line_chart([Series("a", [1, 2], [0, 1])]), target)
        assert target.exists()
        parse(target.read_text(encoding="utf-8"))

    def test_rendering_is_deterministic(self):
        """Figures are committed, so an unchanged experiment must produce an
        unchanged file -- otherwise every run shows up as a spurious diff."""
        args = ([Series("a", [1, 2], [0.3, 0.7])],)
        assert line_chart(*args) == line_chart(*args)
