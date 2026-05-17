"""
Deterministic SVG renderer for scientific figure generation.

Converts JSON layout plans (from the Plan stage) into clean, semantic,
self-contained SVG with grouped editable elements. No external dependencies.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


class SVGRenderer:
    """Converts JSON layout plans to semantic SVG with grouped editable elements.

    Each node, edge, and group in the plan becomes a uniquely-identified
    ``<g>`` element in the SVG, making downstream editing straightforward.
    All text is rendered via ``<text>`` elements (never as paths/pixels), which
    eliminates the text-garbling problem common in generative approaches.
    """

    # ------------------------------------------------------------------
    # Default styling constants
    # ------------------------------------------------------------------
    DEFAULT_FONT_FAMILY = "Arial, Helvetica, sans-serif"
    DEFAULT_FONT_SIZE = 13
    SMALL_FONT_SIZE = 11
    TITLE_FONT_SIZE = 20
    GROUP_LABEL_FONT_SIZE = 12
    LEGEND_FONT_SIZE = 10

    NODE_LABEL_MAX_CHARS = 28  # wrap / truncate labels longer than this

    # ------------------------------------------------------------------
    # Node shape generators
    # ------------------------------------------------------------------
    # Each method on the class whose name matches a node ``type`` value is
    # called automatically.  Shapes supported out-of-the-box:
    #   box, rounded_box, cylinder, diamond, circle, parallelogram, star

    @staticmethod
    def _wrap_label(text: str, max_chars: int) -> List[str]:
        """Break *text* into lines so each line is <= *max_chars* chars."""
        text = text.strip()
        if len(text) <= max_chars:
            return [text]
        # Try breaking on word boundaries; fall back to hard break.
        lines: List[str] = []
        remaining = text
        while len(remaining) > max_chars:
            # Look for last space within limit
            chunk = remaining[:max_chars]
            space = chunk.rfind(" ")
            if space > max_chars // 2:
                lines.append(remaining[:space].strip())
                remaining = remaining[space + 1:]
            else:
                lines.append(remaining[:max_chars])
                remaining = remaining[max_chars:]
        if remaining:
            lines.append(remaining.strip())
        return lines

    @staticmethod
    def _escape_xml(text: str) -> str:
        """Minimal XML-escape for text that will appear in SVG."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    # -- shape primitives ---------------------------------------------------

    def _render_box(
        self, x: float, y: float, w: float, h: float, fill: str, stroke: str
    ) -> str:
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="0" ry="0" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )

    def _render_rounded_box(
        self, x: float, y: float, w: float, h: float, fill: str, stroke: str
    ) -> str:
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="8" ry="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )

    def _render_cylinder(
        self, x: float, y: float, w: float, h: float, fill: str, stroke: str
    ) -> str:
        """Cylinder: top ellipse + vertical sides + bottom arc."""
        cx = x + w / 2
        rx = w / 2
        ry = max(8, h * 0.15)  # ellipse height
        top_ellipse = (
            f'<ellipse cx="{cx}" cy="{y + ry}" rx="{rx}" ry="{ry}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        body = (
            f'<path d="M {x} {y + ry} L {x} {y + h - ry} '
            f'A {rx} {ry} 0 0 0 {x + w} {y + h - ry} '
            f'L {x + w} {y + ry} Z" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        # Bottom arc (dashed to suggest hidden line) and top re-stroke
        bottom_arc = (
            f'<path d="M {x} {y + h - ry} '
            f'A {rx} {ry} 0 0 0 {x + w} {y + h - ry}" '
            f'fill="none" stroke="{stroke}" stroke-width="2" stroke-dasharray="4,3"/>'
        )
        top_restroke = (
            f'<path d="M {x} {y + ry} '
            f'A {rx} {ry} 0 0 0 {x + w} {y + ry}" '
            f'fill="none" stroke="{stroke}" stroke-width="2"/>'
        )
        return f"{body}\n{top_ellipse}\n{bottom_arc}\n{top_restroke}"

    def _render_diamond(
        self, x: float, y: float, w: float, h: float, fill: str, stroke: str
    ) -> str:
        cx, cy = x + w / 2, y + h / 2
        pts = f"{cx},{y} {x + w},{cy} {cx},{y + h} {x},{cy}"
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'

    def _render_circle(
        self, x: float, y: float, w: float, h: float, fill: str, stroke: str
    ) -> str:
        rx, ry = w / 2, h / 2
        return (
            f'<ellipse cx="{x + rx}" cy="{y + ry}" rx="{rx}" ry="{ry}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )

    def _render_parallelogram(
        self, x: float, y: float, w: float, h: float, fill: str, stroke: str
    ) -> str:
        skew = min(20, w * 0.15)
        pts = (
            f"{x + skew},{y} {x + w},{y} "
            f"{x + w - skew},{y + h} {x},{y + h}"
        )
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'

    def _render_star(
        self, x: float, y: float, w: float, h: float, fill: str, stroke: str
    ) -> str:
        """Five-pointed star inscribed in the bounding box."""
        cx, cy = x + w / 2, y + h / 2
        outer_r = min(w, h) / 2
        inner_r = outer_r * 0.4
        points = []
        for i in range(10):
            angle = math.radians(-90 + i * 36)  # start at top
            r = outer_r if i % 2 == 0 else inner_r
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            points.append(f"{px:.2f},{py:.2f}")
        pts = " ".join(points)
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'

    # Shape dispatch --------------------------------------------------------

    def _shape_renderer(self, node_type: str):
        """Return the bound method for the given node type, or fall back to box."""
        method_name = f"_render_{node_type}"
        if hasattr(self, method_name):
            return getattr(self, method_name)
        # Fallback
        return self._render_box

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        plan: dict,
        canvas_width: int = 900,
        canvas_height: int = 700,
    ) -> str:
        """Convert a plan JSON dict to a complete, self-contained SVG string.

        Parameters
        ----------
        plan : dict
            The layout plan produced by the Plan stage.  Must include at least
            ``nodes``; may include ``edges``, ``groups``, and ``title``.
        canvas_width : int
            Preferred canvas width in px.  May be enlarged if nodes overflow.
        canvas_height : int
            Preferred canvas height in px.

        Returns
        -------
        str
            Complete SVG document as a string.
        """
        nodes: List[dict] = plan.get("nodes", [])
        edges: List[dict] = plan.get("edges", [])
        groups: List[dict] = plan.get("groups", [])
        title: str = plan.get("title", "Figure")

        # Build a lookup for nodes by id
        node_lookup: Dict[str, dict] = {n["id"]: n for n in nodes}

        # Auto-size canvas if nodes extend beyond defaults
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        # Compute initial bounds
        for node in nodes:
            nx, ny = node.get("x", 0), node.get("y", 0)
            nw = node.get("width", 140)
            nh = node.get("height", 50)
            min_x = min(min_x, nx)
            min_y = min(min_y, ny)
            max_x = max(max_x, nx + nw)
            max_y = max(max_y, ny + nh)

        # Add groups bounds
        for grp in groups:
            b = grp.get("bounds", {})
            if b:
                min_x = min(min_x, b.get("x", 0))
                min_y = min(min_y, b.get("y", 0))
                max_x = max(max_x, b.get("x", 0) + b.get("width", 0))
                max_y = max(max_y, b.get("y", 0) + b.get("height", 0))

        padding = 60
        title_height = 50
        legend_height = 40 if nodes else 0
        footer = 30

        # Enforce minimum canvas size
        if max_x == float("-inf"):
            max_x, max_y = canvas_width, canvas_height
            min_x, min_y = 0, 0

        # Shift everything so the smallest coordinate lands at (padding, padding+title)
        offset_x = padding - min_x
        offset_y = padding + title_height - min_y

        # Compute final canvas size (at least the requested dimensions)
        needed_w = int(max_x - min_x + 2 * padding)
        needed_h = int(max_y - min_y + 2 * padding + title_height + legend_height + footer)

        final_w = max(canvas_width, needed_w)
        final_h = max(canvas_height, needed_h)

        svg_parts: List[str] = []
        svg_parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {final_w} {final_h}" width="{final_w}" height="{final_h}">'
        )

        # ---- CSS ----------------------------------------------------------
        svg_parts.append("<style>")
        svg_parts.append(
            f".fig-node-label {{ font-family: {self.DEFAULT_FONT_FAMILY}; "
            f"font-size: {self.DEFAULT_FONT_SIZE}px; fill: #1e293b; "
            f"text-anchor: middle; dominant-baseline: central; pointer-events: none; }}"
        )
        svg_parts.append(
            f".fig-edge-label {{ font-family: {self.DEFAULT_FONT_FAMILY}; "
            f"font-size: {self.SMALL_FONT_SIZE}px; fill: #475569; "
            f"text-anchor: middle; dominant-baseline: central; }}"
        )
        svg_parts.append(
            f".fig-edge-label-bg {{ fill: #ffffff; fill-opacity: 0.85; }}"
        )
        svg_parts.append(
            f".fig-group-label {{ font-family: {self.DEFAULT_FONT_FAMILY}; "
            f"font-size: {self.GROUP_LABEL_FONT_SIZE}px; fill: #64748b; "
            f"font-weight: bold; }}"
        )
        svg_parts.append(
            f".fig-title {{ font-family: {self.DEFAULT_FONT_FAMILY}; "
            f"font-size: {self.TITLE_FONT_SIZE}px; fill: #0f172a; "
            f"text-anchor: middle; font-weight: bold; }}"
        )
        svg_parts.append(
            f".fig-legend-text {{ font-family: {self.DEFAULT_FONT_FAMILY}; "
            f"font-size: {self.LEGEND_FONT_SIZE}px; fill: #475569; }}"
        )
        svg_parts.append("</style>")

        # ---- Defs (arrow markers) -----------------------------------------
        svg_parts.append("<defs>")
        # Forward arrow
        svg_parts.append(
            '<marker id="arrow-forward" markerWidth="10" markerHeight="7" '
            'refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">'
            '<polygon points="0 0, 10 3.5, 0 7" fill="#475569"/>'
            "</marker>"
        )
        # Backward arrow
        svg_parts.append(
            '<marker id="arrow-backward" markerWidth="10" markerHeight="7" '
            'refX="1" refY="3.5" orient="auto" markerUnits="strokeWidth">'
            '<polygon points="10 0, 0 3.5, 10 7" fill="#475569"/>'
            "</marker>"
        )
        # Dashed edge arrow
        svg_parts.append(
            '<marker id="arrow-dashed" markerWidth="10" markerHeight="7" '
            'refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">'
            '<polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8"/>'
            "</marker>"
        )
        svg_parts.append("</defs>")

        # ---- Title --------------------------------------------------------
        svg_parts.append(
            f'<text x="{final_w / 2}" y="{padding / 2 + title_height / 2}" '
            f'class="fig-title">{self._escape_xml(title)}</text>'
        )

        # ---- Groups (rendered behind nodes) --------------------------------
        for grp in groups:
            svg_parts.append(
                self._render_group(grp, offset_x, offset_y)
            )

        # ---- Edges (rendered before nodes so they appear underneath) ------
        for edge in edges:
            from_id = edge.get("from", "")
            to_id = edge.get("to", "")
            from_node = node_lookup.get(from_id)
            to_node = node_lookup.get(to_id)
            if from_node and to_node:
                svg_parts.append(
                    self._render_edge(edge, from_node, to_node, offset_x, offset_y)
                )

        # ---- Nodes --------------------------------------------------------
        for node in nodes:
            svg_parts.append(
                self.render_node(node, offset_x, offset_y)
            )

        # ---- Legend -------------------------------------------------------
        if nodes:
            svg_parts.append(self._render_legend(final_w, final_h))

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    # ------------------------------------------------------------------
    # Node rendering
    # ------------------------------------------------------------------

    def render_node(self, node: dict, offset_x: float = 0, offset_y: float = 0) -> str:
        """Render a single node as an SVG ``<g>`` group with shape + label text.

        Parameters
        ----------
        node : dict
            Node definition with id, label, type, x, y, width, height, color.
        offset_x : float
            Horizontal offset applied to all coordinates.
        offset_y : float
            Vertical offset applied to all coordinates.

        Returns
        -------
        str
            SVG ``<g>`` element string.
        """
        node_id = self._escape_xml(node.get("id", "?"))
        label = node.get("label", node_id)
        node_type = node.get("type", "box")
        x = node.get("x", 0) + offset_x
        y = node.get("y", 0) + offset_y
        w = node.get("width", 140)
        h = node.get("height", 50)
        fill = node.get("color", "#e8f4fd")
        stroke = "#1e40af"  # darker blue for border

        # Render shape
        render_fn = self._shape_renderer(node_type)
        shape_svg = render_fn(x, y, w, h, fill, stroke)

        # Render label text (possibly multi-line)
        lines = self._wrap_label(label, self.NODE_LABEL_MAX_CHARS)
        text_elements: List[str] = []
        cx = x + w / 2
        cy = y + h / 2
        line_height = self.DEFAULT_FONT_SIZE + 2
        start_y = cy - (len(lines) - 1) * line_height / 2

        for i, line in enumerate(lines):
            ly = start_y + i * line_height
            text_elements.append(
                f'<text x="{cx}" y="{ly}" class="fig-node-label">'
                f"{self._escape_xml(line)}</text>"
            )

        parts = [
            f'<!-- node: {node_id} -->',
            f'<g id="node-{node_id}">',
            shape_svg,
            *text_elements,
            "</g>",
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Edge rendering
    # ------------------------------------------------------------------

    def _render_edge(
        self,
        edge: dict,
        from_node: dict,
        to_node: dict,
        offset_x: float,
        offset_y: float,
    ) -> str:
        """Render an edge as SVG ``<g>`` with path, arrowhead(s), and label.

        Calculates start/end points based on node positions, taking the
        closest point on each node's bounding box as the connection anchor.
        """
        from_id = edge.get("from", "?")
        to_id = edge.get("to", "?")
        label = edge.get("label", "")
        style = edge.get("style", "solid")
        direction = edge.get("direction", "forward")

        # Extract node geometry (with offset)
        fx = from_node.get("x", 0) + offset_x
        fy = from_node.get("y", 0) + offset_y
        fw = from_node.get("width", 140)
        fh = from_node.get("height", 50)

        tx = to_node.get("x", 0) + offset_x
        ty = to_node.get("y", 0) + offset_y
        tw = to_node.get("width", 140)
        th = to_node.get("height", 50)

        # Center points
        from_cx = fx + fw / 2
        from_cy = fy + fh / 2
        to_cx = tx + tw / 2
        to_cy = ty + th / 2

        # Find the connection points on the perimeters
        p1 = self._closest_rect_point(from_cx, from_cy, fx, fy, fw, fh, to_cx, to_cy)
        p2 = self._closest_rect_point(to_cx, to_cy, tx, ty, tw, th, from_cx, from_cy)

        # Determine stroke dasharray and marker
        if style == "dashed":
            dash = "8,4"
            marker_end = "url(#arrow-dashed)"
            stroke_color = "#94a3b8"
        elif style == "dotted":
            dash = "3,3"
            marker_end = "none"
            stroke_color = "#94a3b8"
        else:  # solid
            dash = "none"
            marker_end = "url(#arrow-forward)"
            stroke_color = "#475569"

        marker_start = "none"
        if direction == "bidirectional":
            marker_start = "url(#arrow-backward)"
        elif direction == "backward":
            # Swap: arrow at start, nothing at end
            marker_start = marker_end
            marker_end = "none"

        # Build path
        path_d = f"M {p1[0]:.1f} {p1[1]:.1f} L {p2[0]:.1f} {p2[1]:.1f}"

        parts = [
            f'<!-- edge: {from_id} -> {to_id} -->',
            f'<g id="edge-{from_id}-{to_id}">',
            f'<path d="{path_d}" fill="none" stroke="{stroke_color}" '
            f'stroke-width="2" stroke-dasharray="{dash}" '
            f'marker-start="{marker_start}" marker-end="{marker_end}"/>',
        ]

        # Edge label at midpoint with white background
        if label:
            mx = (p1[0] + p2[0]) / 2
            my = (p1[1] + p2[1]) / 2
            label_esc = self._escape_xml(label)
            # Estimate text width for background rect (rough: 7px per char)
            est_w = max(40, len(label) * 7 + 10)
            est_h = 18
            parts.append(
                f'<rect x="{mx - est_w / 2}" y="{my - est_h / 2}" '
                f'width="{est_w}" height="{est_h}" class="fig-edge-label-bg" rx="3"/>'
            )
            parts.append(
                f'<text x="{mx}" y="{my}" class="fig-edge-label">{label_esc}</text>'
            )

        parts.append("</g>")
        return "\n".join(parts)

    @staticmethod
    def _closest_rect_point(
        cx: float, cy: float,
        rx: float, ry: float, rw: float, rh: float,
        target_x: float, target_y: float,
    ) -> Tuple[float, float]:
        """Return the point on the perimeter of rectangle (rx,ry,rw,rh) closest
        to (target_x, target_y), when the line starts from (cx, cy) which is
        the rectangle center."""
        # Clamp target to rectangle; the intersection with the perimeter gives
        # the anchor point.  This is a classic "line-rect intersection" from
        # center outward.
        dx = target_x - cx
        dy = target_y - cy

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            # Degenerate: just return center right edge
            return (rx + rw, cy)

        half_w = rw / 2
        half_h = rh / 2

        # Scale factor to hit the rectangle boundary
        scale_x = half_w / abs(dx) if abs(dx) > 1e-9 else float("inf")
        scale_y = half_h / abs(dy) if abs(dy) > 1e-9 else float("inf")
        scale = min(scale_x, scale_y)

        px = cx + dx * scale
        py = cy + dy * scale

        # Clamp to rectangle bounds (floating point edge cases)
        px = max(rx, min(rx + rw, px))
        py = max(ry, min(ry + rh, py))

        return (px, py)

    # ------------------------------------------------------------------
    # Group rendering
    # ------------------------------------------------------------------

    def _render_group(
        self, group: dict, offset_x: float, offset_y: float
    ) -> str:
        """Render a bounding box group with dashed border and label."""
        label = group.get("label", "Group")
        bounds = group.get("bounds", {})
        node_ids = group.get("nodes", [])

        x = bounds.get("x", 0) + offset_x
        y = bounds.get("y", 0) + offset_y
        w = bounds.get("width", 200)
        h = bounds.get("height", 100)
        color = group.get("color", "#f8fafc")
        stroke = "#cbd5e1"

        group_id = self._escape_xml(label.lower().replace(" ", "-"))

        parts = [
            f'<!-- group: {self._escape_xml(label)} -->',
            f'<g id="group-{group_id}">',
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="12" fill="{color}" fill-opacity="0.4" '
            f'stroke="{stroke}" stroke-width="1.5" stroke-dasharray="8,4"/>',
            f'<text x="{x + 12}" y="{y + 18}" class="fig-group-label">'
            f"{self._escape_xml(label)}</text>",
            "</g>",
        ]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    def _render_legend(self, canvas_w: float, canvas_h: float) -> str:
        """Render a simple legend strip at the bottom showing the node types used."""
        # Collect unique types from the current render call — we store them
        # at render time.  Because _render_legend is called from render() we
        # can pass them via a parameter.  For simplicity, we show a fixed set
        # of common shape examples.
        legend_y = canvas_h - 25
        x_start = canvas_w / 2 - 200

        sample_shapes = [
            ("box", "Box"),
            ("rounded_box", "Rounded"),
            ("cylinder", "Cylinder"),
            ("diamond", "Diamond"),
            ("circle", "Circle"),
            ("parallelogram", "Para."),
            ("star", "Star"),
        ]

        parts = [
            '<g id="legend">',
        ]

        spacing = 62
        for i, (stype, slabel) in enumerate(sample_shapes):
            sx = x_start + i * spacing
            sy = legend_y - 6
            # Tiny shape preview
            render_fn = self._shape_renderer(stype)
            shape_str = render_fn(sx, sy, 18, 12, "#e2e8f0", "#94a3b8")
            parts.append(shape_str)
            parts.append(
                f'<text x="{sx + 22}" y="{sy + 6}" class="fig-legend-text">'
                f"{slabel}</text>"
            )

        parts.append("</g>")
        return "\n".join(parts)
