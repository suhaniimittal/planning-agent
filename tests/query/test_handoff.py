"""The agreed TDD input as the planner's text (2026-10-29): the card, its
ranked services and the review's questions with their answers."""

from src.query import handoff

PAYLOAD = {
    "card": {
        "key": "BTS-60",
        "type": "Story",
        "title": "Add a remark to IMPS payments",
        "description": "Customers should be able to add a remark when sending an IMPS payment.",
        "comments": ["Please keep NEFT as it is."],
        "files": [
            {"name": "spec.pdf", "content": "POST /imps/transfer"},
            {"name": "recording.mp4", "content": ""},
        ],
    },
    "services": [
        {
            "service": "statement-service",
            "repo": "AetherionAgentDemo/bank",
            "branch": "main",
            "rank": 7,
            "reason": "The remark must appear on the statement",
            "prs": [],
        },
        {
            "service": "imps-service",
            "repo": "AetherionAgentDemo/bank",
            "branch": "main",
            "rank": 9,
            "reason": "spec.pdf names POST /imps/transfer",
            "prs": [],
        },
        {
            "service": "payments-service",
            "repo": "AetherionAgentDemo/bank",
            "branch": "main",
            "rank": 8,
            "reason": "RemarkValidator already checks remark length",
            "prs": [
                {
                    "card": "BTS-20",
                    "pr": 41,
                    "branch": "feature/BTS-20-neft-remark",
                    "url": "https://github.com/AetherionAgentDemo/bank/pull/41",
                }
            ],
        },
    ],
    "questions": [
        {
            "question": "Does the 50-character rule apply to IMPS remarks?",
            "answers": ["Yes, same as NEFT."],
        },
        {"question": "Should the SMS confirmation include the remark?", "answers": []},
    ],
}


def test_the_planner_reads_the_card_its_ranked_services_and_the_answers():
    text = handoff.issue_text(PAYLOAD)
    assert text.startswith("BTS-60 (Story): Add a remark to IMPS payments")
    assert "=== spec.pdf ===\nPOST /imps/transfer" in text
    assert "=== recording.mp4 === (not read)" in text
    assert "- Please keep NEFT as it is." in text
    order = [
        text.index(f"- {s}") for s in ("imps-service", "payments-service", "statement-service")
    ]
    assert order == sorted(order)
    assert "earlier: BTS-20 PR #41 on branch feature/BTS-20-neft-remark" in text
    assert "Does the 50-character rule apply to IMPS remarks?\n  Answer: Yes, same as NEFT." in text
    open_part = text[text.index("Open questions") :]
    assert "Should the SMS confirmation include the remark?" in open_part


def test_every_ranked_service_is_named_so_the_planner_designs_it():
    from src.query import code_search

    text = handoff.issue_text(PAYLOAD)
    named = code_search.services_named_in(
        text, ["imps-service", "payments-service", "statement-service", "audit-service"]
    )
    assert named == ["imps-service", "payments-service", "statement-service"]


def test_the_repositories_are_the_ranked_services():
    assert handoff.repos_text(PAYLOAD) == (
        "statement-service: AetherionAgentDemo/bank@main\n"
        "imps-service: AetherionAgentDemo/bank@main\n"
        "payments-service: AetherionAgentDemo/bank@main"
    )
