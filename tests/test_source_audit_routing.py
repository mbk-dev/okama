"""Check the source audit selected for each supported GitHub Actions event."""

import re
import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/security-audit.yml"
ACTION = "mbk-dev/cbrapi/.github/actions/security-audit@"
GITLEAKS_PIN = "2b32319a99d4da63644995eed3faf18faec28cea"
FULL_PIN = "08be62bd91bcc85c04605889701a4bcd34e25a07"


def selected_audits(workflow: str, event: str) -> list[tuple[str, str]]:
    """Evaluate the workflow's event guards for the pinned audit steps."""
    selected = []
    for step in re.split(r"^      - ", workflow, flags=re.MULTILINE)[1:]:
        action = re.search(rf"^        uses: {re.escape(ACTION)}([0-9a-f]{{40}})$", step, re.M)
        if not action:
            continue

        condition = re.search(r"^        if: (.+)$", step, re.M)
        if condition:
            expression = re.fullmatch(r"\$\{\{\s*(!?)\(([^()]*)\)\s*\}\}", condition[1])
            if not expression:
                raise AssertionError(f"Unsupported source audit condition: {condition[1]}")
            negated, alternatives = expression.groups()
            events = []
            for term in alternatives.split(" || "):
                predicate = re.fullmatch(r"github\.event_name == '([a-z_]+)'", term)
                if not predicate:
                    raise AssertionError(f"Unsupported source audit predicate: {term}")
                events.append(predicate[1])
            if (event in events) == bool(negated):
                continue

        stage = re.search(r"^          stage: ([a-z]+)$", step, re.M)
        selected.append((action[1], stage[1] if stage else "all"))
    return selected


class TestSourceAuditRouting(unittest.TestCase):
    def test_automatic_and_full_audits_select_exclusive_stages(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        cases = {
            "pull_request": [(GITLEAKS_PIN, "secrets")],
            "push": [(GITLEAKS_PIN, "secrets")],
            "workflow_dispatch": [(FULL_PIN, "all")],
            "schedule": [(FULL_PIN, "all")],
        }
        for event, expected in cases.items():
            with self.subTest(event=event):
                self.assertEqual(selected_audits(workflow, event), expected)


if __name__ == "__main__":
    unittest.main()
