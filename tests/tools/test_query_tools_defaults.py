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


def test_the_form_does_not_offer_top_k_or_hops():
    """Deliberately not on the form: both are retrieval tuning, not something
    a person filing a ticket should have to answer, and the help text beside
    them went stale the moment the defaults changed. Leaving them out is what
    makes build_tdd's signature the only place they are set."""
    triggers = json.loads((_ROOT / "src" / "agent" / "metadata.json").read_text())["config"][
        "triggers"
    ]
    assert [t["name"] for t in triggers] == ["issue_text", "repos"]
