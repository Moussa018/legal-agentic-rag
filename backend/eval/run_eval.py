"""Compare vector-only vs hybrid on multi-hop questions.

Run from backend/: python -m eval.run_eval
Metric: a question passes if every expected string appears in the answer (case-insensitive).
Replace questions.json with questions + gold answers built from YOUR corpus.
"""
import json
from pathlib import Path

from app.agent import run_agent
from app.qa import answer


def ask(question: str, mode: str) -> dict:
    return run_agent(question) if mode == "agentic" else answer(question, mode)


def passes(ans: str, expected: list[str]) -> bool:
    low = ans.lower()
    return all(e.lower() in low for e in expected)


def main() -> None:
    questions = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
    score = {"vector": 0, "hybrid": 0, "agentic": 0}
    for q in questions:
        for mode in score:
            if passes(ask(q["question"], mode)["answer"], q["expected"]):
                score[mode] += 1
    n = len(questions)
    for mode, s in score.items():
        print(f"{mode:7s} {s}/{n} = {100 * s / n:.0f}%")


if __name__ == "__main__":
    main()
