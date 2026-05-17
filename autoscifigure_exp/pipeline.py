"""
Automated scientific figure generation pipeline.

Four-stage pipeline:
  1. Plan  (LLM)  -- text -> JSON layout plan
  2. Render (det.)  -- JSON plan -> semantic SVG
  3. Critique (VLM) -- SVG + original text -> structured evaluation
  4. Refine (LLM)  -- fix issues -> re-render -> re-critique (loop)

Uses the Anthropic Claude API for all LLM/VLM calls.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import anthropic

# Import our sibling renderer
# (Allow running from any directory as long as renderer.py is importable.)
try:
    from renderer import SVGRenderer
except ImportError:
    # Fallback: add the directory containing *this* file to sys.path
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from renderer import SVGRenderer  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert scientific figure designer. Your task is to convert a
research paper excerpt into a clean, accurate JSON layout plan for an SVG
diagram.  You MUST output ONLY valid JSON — no preamble, no markdown fences,
no commentary.  The JSON must match the schema described below.

You are designing a figure that represents the key entities and relationships
described in the text.  Choose the most appropriate figure_type, position
nodes logically (left-to-right or top-to-bottom flow), group related nodes,
and select appropriate shapes for each entity.

Node types and their semantic meanings:
  - box:          generic component / module
  - rounded_box:  process / function / algorithm
  - cylinder:     database / data store / file
  - diamond:      decision / condition
  - circle:       start/end / external entity
  - parallelogram: input/output data
  - star:         emphasis / novel contribution

Edge styles: solid (primary flow), dashed (optional/feedback), dotted (annotation).

Color palette suggestions (use soft pastel tones):
  - Blue tones:  #dbeafe, #bfdbfe, #93c5fd
  - Green tones: #dcfce7, #bbf7d0, #86efac
  - Yellow tones: #fef9c3, #fef08a
  - Purple tones: #f3e8ff, #e9d5ff
  - Gray tones:  #f1f5f9, #e2e8f0

Guidelines:
  - Each node needs an id, label, type, x, y, width, height, color, description.
  - x/y positions should leave ~20px gap between adjacent nodes.
  - Standard node size: width 120-180, height 45-65.
  - Every edge needs from, to, label, style, direction.
  - Use groups to show architectural boundaries (Encoder block, Decoder, etc.).
  - Group bounds must enclose all member nodes with at least 15px padding.
  - Include a descriptive title.
""")

PLAN_USER_TEMPLATE = """\
Convert the following scientific text into a figure layout plan.

TEXT:
{text}

FIGURE TYPE: {figure_type}

Output ONLY the JSON plan (no markdown fences):
"""

CRITIQUE_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert reviewer of scientific figures.  You will be shown a
scientific figure (as SVG markup) alongside the original text it was
generated from.  Your job is to critically evaluate the figure and return
a structured JSON assessment.

Evaluation criteria (score each 0-10):
  - structure_correctness: Does the diagram faithfully capture the entities,
    relationships, and architecture described in the text?  Any missing or
    spurious elements?
  - text_accuracy: Are all labels, titles, and edge annotations spelled
    correctly and matching the text?  Any garbled or hallucinated text?
  - visual_clarity: Is the layout clean and easy to read?  Are nodes well-
    positioned, groups clear, colors appropriate, overlaps avoided?
  - overall_quality: Your holistic judgment.  Weight structure_correctness
    and text_accuracy more heavily.

For each issue you find, provide:
  - severity: high | medium | low
  - category: structure | text | layout | style
  - description: What is wrong
  - location: Which node_id or edge_id is affected (or "overall")

Output ONLY valid JSON — no preamble, no markdown fences.
""")

CRITIQUE_USER_TEMPLATE = """\
Original text:
{text}

SVG figure (markup):
{svg}

Please evaluate the SVG figure against the original text and return your
assessment as JSON:
"""

REFINE_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert scientific figure designer fixing issues in a generated
diagram.  You will receive the original layout plan JSON and a list of
issues found during review.  Your task is to produce a corrected JSON plan
that addresses every issue.

Fix rules:
  - For structure issues: add missing nodes/edges, remove spurious ones,
    correct edge directions.
  - For text issues: fix spelling, terminology, labels to match the original
    paper text exactly.
  - For layout issues: adjust x/y positions to avoid overlaps and improve
    flow.  Increase spacing to at least 25px between nodes.
  - For style issues: adjust colors, fix node types, improve grouping.
  - NEVER remove nodes or edges that are correct — only fix what is broken.
  - Preserve the existing id scheme unless a node must be added or removed.

Output ONLY the corrected JSON plan — no preamble, no markdown fences.
""")

REFINE_USER_TEMPLATE = """\
Current plan JSON:
{plan_json}

Critique issues to fix:
{issues_json}

Original text for reference:
{text}

Output the CORRECTED JSON plan:
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_TEXT_CHARS = 4000


def _truncate_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Truncate a long text input to *max_chars* characters, keeping a
    natural break point when possible."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to break at last paragraph or sentence boundary
    last_para = truncated.rfind("\n\n")
    last_sentence = truncated.rfind(". ")
    cut = max(last_para, last_sentence)
    if cut > max_chars * 0.6:
        return truncated[: cut + 1] + "\n\n[... text truncated ...]"
    return truncated + "\n\n[... text truncated ...]"


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract a JSON object from an LLM response string.

    Handles:
      - Plain JSON
      - JSON inside ```json ... ``` fences
      - JSON inside ``` ... ``` fences
      - Leading/trailing whitespace
    """
    if not text or not text.strip():
        return None

    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    fence_patterns = [
        r"```json\s*([\s\S]*?)```",
        r"```\s*([\s\S]*?)```",
    ]
    for pattern in fence_patterns:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Try to find the outermost { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class SciFigurePipeline:
    """Main pipeline that generates editable SVG scientific figures from text.

    Parameters
    ----------
    api_key : str or None
        Anthropic API key.  If ``None``, reads the ``ANTHROPIC_API_KEY``
        environment variable.
    model : str
        Claude model identifier used for all LLM/VLM calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "DeepSeek-V4-pro",
        base_url: Optional[str] = None,
    ):
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key must be provided or set as "
                "ANTHROPIC_API_KEY environment variable."
            )
        self.api_key = resolved_key
        self.model = model
        resolved_base = base_url or os.environ.get("ANTHROPIC_BASE_URL", None)
        client_kwargs = {"api_key": self.api_key}
        if resolved_base:
            client_kwargs["base_url"] = resolved_base
        self.client = anthropic.Anthropic(**client_kwargs)
        self.renderer = SVGRenderer()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        figure_type: str = "auto",
        max_rounds: int = 3,
    ) -> dict:
        """Generate a scientific figure from paper text.

        Parameters
        ----------
        text : str
            The scientific text to visualize (paper excerpt, abstract, etc.).
            Automatically truncated to ~4000 chars if longer.
        figure_type : str
            Hint for figure type: "auto", "architecture_diagram", "flowchart",
            "model_architecture", "data_pipeline".  "auto" lets the LLM decide.
        max_rounds : int
            Maximum number of refine (critique -> fix) cycles after the initial
            plan.  Default 3.

        Returns
        -------
        dict with keys:
            plan      — the final JSON layout plan
            svg       — the final SVG string
            critique  — the final critique JSON
            rounds    — how many refine rounds were executed
            history   — list of all intermediate (plan, svg, critique) states
        """
        print("=" * 60)
        print("SciFigurePipeline: Starting figure generation")
        print(f"  Model: {self.model}")
        print(f"  Figure type hint: {figure_type}")
        print(f"  Max refine rounds: {max_rounds}")
        print(f"  Input text length: {len(text)} chars")
        print("=" * 60)

        # Truncate long text
        text = _truncate_text(text)

        history: List[dict] = []

        # ---- Stage 1: Plan ------------------------------------------------
        print("\n[Stage 1/4] PLAN: Extracting layout plan from text ...")
        plan = self._plan(text, figure_type)
        print(f"  Plan produced: {len(plan.get('nodes', []))} nodes, "
              f"{len(plan.get('edges', []))} edges, "
              f"{len(plan.get('groups', []))} groups")

        # ---- Stage 2: Render ----------------------------------------------
        print("\n[Stage 2/4] RENDER: Converting plan to SVG ...")
        svg = self._render(plan)
        print(f"  SVG produced: {len(svg)} chars")

        # ---- Stage 3: Critique --------------------------------------------
        print("\n[Stage 3/4] CRITIQUE: Evaluating SVG against original text ...")
        critique = self._critique(text, svg)
        overall = critique.get("overall_quality", 0)
        n_issues = len(critique.get("issues", []))
        print(f"  Critique score: {overall}/10  |  Issues: {n_issues}")

        history.append({
            "stage": "initial",
            "plan": plan,
            "svg": svg,
            "critique": critique,
        })

        # ---- Stage 4: Refine loop -----------------------------------------
        rounds = 0
        while overall < 8.0 and rounds < max_rounds:
            rounds += 1
            print(f"\n[Stage 4/4] REFINE round {rounds}/{max_rounds} "
                  f"(score {overall}/10 < 8.0)")

            # Refine plan
            plan = self._refine(plan, critique, text)
            print(f"  Refined plan: {len(plan.get('nodes', []))} nodes, "
                  f"{len(plan.get('edges', []))} edges")

            # Re-render
            svg = self._render(plan)
            print(f"  Re-rendered SVG: {len(svg)} chars")

            # Re-critique
            critique = self._critique(text, svg)
            overall = critique.get("overall_quality", 0)
            n_issues = len(critique.get("issues", []))
            print(f"  New score: {overall}/10  |  Issues: {n_issues}")

            history.append({
                "stage": f"refine_round_{rounds}",
                "plan": plan,
                "svg": svg,
                "critique": critique,
            })

        if overall >= 8.0:
            print(f"\nTarget quality achieved ({overall}/10 >= 8.0). Done.")
        else:
            print(f"\nMax refine rounds reached. Final score: {overall}/10")

        print("=" * 60)
        print("SciFigurePipeline: Generation complete.")
        print(f"  Rounds executed: {rounds}")
        print(f"  Final score: {overall}/10")
        print("=" * 60)

        return {
            "plan": plan,
            "svg": svg,
            "critique": critique,
            "rounds": rounds,
            "history": history,
        }

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _plan(self, text: str, figure_type: str) -> dict:
        """Stage 1: LLM call to extract a JSON layout plan from text."""
        user_prompt = PLAN_USER_TEMPLATE.format(
            text=text, figure_type=figure_type
        )
        raw = self._call_claude(
            system_prompt=PLAN_SYSTEM_PROMPT,
            user_message=user_prompt,
            max_tokens=4096,
            stage="Plan",
        )
        result = _extract_json(raw)
        if result is None:
            print("  WARNING: Failed to parse Plan JSON. Using fallback plan.")
            return self._fallback_plan(text)
        return result

    def _render(self, plan: dict) -> str:
        """Stage 2: Deterministic conversion of plan JSON to SVG."""
        try:
            return self.renderer.render(plan)
        except Exception as exc:
            print(f"  ERROR during render: {exc}")
            # Return a minimal error SVG
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 200">'
                '<text x="200" y="100" text-anchor="middle" '
                'font-family="Arial" font-size="14" fill="red">'
                f'Render Error: {SVGRenderer._escape_xml(str(exc))}</text>'
                "</svg>"
            )

    def _critique(self, text: str, svg: str) -> dict:
        """Stage 3: VLM call to evaluate SVG against original text.

        The SVG is passed as raw markup inside the user message so Claude
        can "see" the figure structure.
        """
        # Truncate SVG if extremely large (unlikely but defensive)
        svg_for_critique = svg
        max_svg_chars = 30000
        if len(svg) > max_svg_chars:
            svg_for_critique = svg[:max_svg_chars] + "\n<!-- SVG truncated -->"
            print(f"  SVG truncated from {len(svg)} to {max_svg_chars} chars for critique")

        user_prompt = CRITIQUE_USER_TEMPLATE.format(
            text=text, svg=svg_for_critique
        )
        raw = self._call_claude(
            system_prompt=CRITIQUE_SYSTEM_PROMPT,
            user_message=user_prompt,
            max_tokens=2048,
            stage="Critique",
        )
        result = _extract_json(raw)
        if result is None:
            print("  WARNING: Failed to parse Critique JSON. Using default critique.")
            return self._fallback_critique()
        return result

    def _refine(
        self,
        plan: dict,
        critique: dict,
        original_text: str,
    ) -> dict:
        """Stage 4: LLM call to fix issues identified in the critique."""
        issues = critique.get("issues", [])
        if not issues:
            print("  No issues to fix; returning plan unchanged.")
            return plan

        user_prompt = REFINE_USER_TEMPLATE.format(
            plan_json=json.dumps(plan, indent=2, ensure_ascii=False),
            issues_json=json.dumps(issues, indent=2, ensure_ascii=False),
            text=original_text,
        )
        raw = self._call_claude(
            system_prompt=REFINE_SYSTEM_PROMPT,
            user_message=user_prompt,
            max_tokens=4096,
            stage="Refine",
        )
        result = _extract_json(raw)
        if result is None:
            print("  WARNING: Failed to parse Refine JSON. Returning original plan.")
            return plan
        return result

    # ------------------------------------------------------------------
    # Claude API helper
    # ------------------------------------------------------------------

    def _call_claude(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
        stage: str = "",
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """Call the Anthropic Claude API with retry logic.

        Parameters
        ----------
        system_prompt : str
            System-level instruction.
        user_message : str
            The user turn content.
        max_tokens : int
            Maximum tokens in the response.
        stage : str
            Label used in log messages.
        retries : int
            Number of retries on transient errors.
        retry_delay : float
            Base delay between retries (doubles each attempt).

        Returns
        -------
        str
            The text content of Claude's response.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message},
                    ],
                )
                # Extract text from the response content block(s)
                text_parts: List[str] = []
                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                return "".join(text_parts)

            except anthropic.APIError as exc:
                last_error = exc
                print(f"  [{stage}] API error (attempt {attempt}/{retries}): {exc}")
            except anthropic.RateLimitError as exc:
                last_error = exc
                wait = retry_delay * (2 ** (attempt - 1))
                print(f"  [{stage}] Rate limited (attempt {attempt}/{retries}); "
                      f"waiting {wait:.1f}s ...")
                time.sleep(wait)
            except anthropic.APIConnectionError as exc:
                last_error = exc
                wait = retry_delay * (2 ** (attempt - 1))
                print(f"  [{stage}] Connection error (attempt {attempt}/{retries}); "
                      f"waiting {wait:.1f}s ...")
                time.sleep(wait)
            except Exception as exc:
                last_error = exc
                print(f"  [{stage}] Unexpected error (attempt {attempt}/{retries}): {exc}")
                time.sleep(retry_delay)

        raise RuntimeError(
            f"[{stage}] All {retries} API call attempts failed. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Fallback / defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_plan(text: str) -> dict:
        """Return a minimal valid plan when JSON parsing fails."""
        title = (text[:60].strip() + "...") if len(text) > 60 else text.strip()
        return {
            "title": title,
            "figure_type": "architecture_diagram",
            "nodes": [
                {
                    "id": "n1",
                    "label": "Input",
                    "type": "parallelogram",
                    "x": 50,
                    "y": 100,
                    "width": 140,
                    "height": 50,
                    "color": "#dbeafe",
                    "description": "Input data / text",
                },
                {
                    "id": "n2",
                    "label": "Process",
                    "type": "rounded_box",
                    "x": 280,
                    "y": 100,
                    "width": 140,
                    "height": 50,
                    "color": "#dcfce7",
                    "description": "Main processing step",
                },
                {
                    "id": "n3",
                    "label": "Output",
                    "type": "parallelogram",
                    "x": 510,
                    "y": 100,
                    "width": 140,
                    "height": 50,
                    "color": "#fef9c3",
                    "description": "Output result",
                },
            ],
            "edges": [
                {
                    "from": "n1",
                    "to": "n2",
                    "label": "data flow",
                    "style": "solid",
                    "direction": "forward",
                },
                {
                    "from": "n2",
                    "to": "n3",
                    "label": "data flow",
                    "style": "solid",
                    "direction": "forward",
                },
            ],
            "groups": [],
        }

    @staticmethod
    def _fallback_critique() -> dict:
        """Return a neutral critique when JSON parsing fails."""
        return {
            "structure_correctness": 7,
            "text_accuracy": 7,
            "visual_clarity": 7,
            "overall_quality": 7,
            "issues": [
                {
                    "severity": "medium",
                    "category": "structure",
                    "description": "Automatic critique unavailable — JSON parse failed.",
                    "location": "overall",
                }
            ],
            "summary": "Critique JSON could not be parsed; using default scores.",
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def generate_figure(
    text: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-6",
    figure_type: str = "auto",
    max_rounds: int = 3,
) -> dict:
    """One-shot convenience wrapper around ``SciFigurePipeline.generate``.

    Parameters
    ----------
    text : str
        Scientific text to visualize.
    api_key : str or None
        Anthropic API key.
    model : str
        Claude model identifier.
    figure_type : str
        Figure type hint.
    max_rounds : int
        Max refine rounds.

    Returns
    -------
    dict
        Same structure as ``SciFigurePipeline.generate``.
    """
    pipeline = SciFigurePipeline(api_key=api_key, model=model)
    return pipeline.generate(
        text=text,
        figure_type=figure_type,
        max_rounds=max_rounds,
    )


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly:  python pipeline.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "We propose a Transformer-based architecture for text-to-SQL "
        "generation. The system consists of three main components: a "
        "Schema Encoder that embeds database schema into a latent "
        "representation, a Question Encoder that tokenizes and encodes "
        "natural language questions, and a Decoder that generates SQL "
        "queries by attending over both encodings. The Schema Encoder "
        "outputs are stored in a Schema Memory module. The Decoder "
        "interacts with a Database Executor to validate generated SQL."
    )

    print("SciFigurePipeline smoke test")
    print("=" * 60)
    print("This will attempt to call the Anthropic API.\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set.  Set it to run the smoke test.")
        print("Alternatively, run in Python and pass api_key explicitly.")
        sys.exit(1)

    result = generate_figure(
        text=sample_text,
        api_key=api_key,
        figure_type="model_architecture",
        max_rounds=2,
    )

    print("\nFinal plan JSON:")
    print(json.dumps(result["plan"], indent=2, ensure_ascii=False))

    print("\nFinal critique:")
    print(json.dumps(result["critique"], indent=2, ensure_ascii=False))

    print(f"\nSVG length: {len(result['svg'])} chars")
    print(f"Rounds: {result['rounds']}")
    print(f"History entries: {len(result['history'])}")

    # Optionally write SVG to file
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output.svg")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result["svg"])
    print(f"\nSVG written to: {out_path}")
