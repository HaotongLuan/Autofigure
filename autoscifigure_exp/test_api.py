"""Quick test of pipeline with DeepSeek API."""
import os, sys
os.environ["ANTHROPIC_API_KEY"] = "sk-5832884ed915490988b3e88ad4a45bc9"
os.environ["ANTHROPIC_BASE_URL"] = "https://api.deepseek.com/anthropic"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import SciFigurePipeline
from test_cases import get_test_case

pipeline = SciFigurePipeline()
tc = get_test_case(["tc01"])[0]
print(f"Test: {tc['id']} - {tc['title']}")
print(f"Text length: {len(tc['text'])} chars")
print("Running Plan stage...")

result = pipeline.generate(tc["text"], tc["figure_type"], max_rounds=1)
print(f"Plan keys: {list(result['plan'].keys())}")
print(f"Nodes: {len(result['plan']['nodes'])}")
print(f"Edges: {len(result['plan']['edges'])}")
print(f"Groups: {len(result['plan']['groups'])}")
print(f"SVG length: {len(result['svg'])} chars")
cq = result.get("critique", {})
print(f"Critique overall_quality: {cq.get('overall_quality', 'N/A')}")
print(f"Rounds: {result['rounds']}")

# Save
with open("results/test_tc01.svg", "w", encoding="utf-8") as f:
    f.write(result["svg"])
print("Saved SVG. SUCCESS!")
