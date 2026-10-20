import inspect
import json
from pathlib import Path

from src.query.query_flow import build_tdd
from src.tools.query_tools import generate_technical_design

_ROOT = Path(__file__).resolve().parents[2]


def _default(fn, name):
    return inspect.signature(fn).parameters[name].default


def test_the_tool_defers_to_build_tdd_for_top_k_and_hops():
    """Both were duplicated in the agent workflow, which kept passing 5 and 2
    long after the real defaults changed — so a form left blank silently
    ignored them. Nothing outside build_tdd may name a number."""
    for name in ("top_k", "hops"):
        assert _default(generate_technical_design, name) is None, name


def test_the_agent_workflow_names_no_default_of_its_own():
    source = (_ROOT / "src" / "agent" / "agent.py").read_text()
    for forbidden in ('payload.get("top_k") or', "if hops is None else"):
        assert forbidden not in source, forbidden


def test_the_form_help_text_matches_the_real_defaults():
    """The numbers users read in the Aetherion form live in metadata.json and
    drifted out of step with the code."""
    triggers = json.loads((_ROOT / "src" / "agent" / "metadata.json").read_text())["config"][
        "triggers"
    ]
    described = {t["name"]: t["description"] for t in triggers}
    assert f"({_default(build_tdd, 'top_k')})" in described["top_k"]
    assert f"({_default(build_tdd, 'hops')}, off)" in described["hops"]
