"""One agent's revision turn: the messages it sees, and which answer its revision commits to."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from conf_compose.constants import TASKS

from .prompts import PEER, PEER_LABELS, REVISE, answer_format, intro

_NAMES = "|".join(PEER_LABELS)
# Agreement that names a peer: "I agree with Beta", "Beta's answer is correct", "Beta is right".
_AGREEMENT = (
    re.compile(rf"\b(?:agree|concur|side|go|align)s?\s+with\s+(?:agent\s+)?({_NAMES})\b", re.IGNORECASE),
    re.compile(rf"\b({_NAMES})(?:'s)?\s+(?:final\s+)?(?:answer|reasoning|solution|conclusion)\s+"
               rf"(?:is|was|seems|appears)\s+(?:to\s+be\s+)?(?:correct|right|sound|accurate)", re.IGNORECASE),
    re.compile(rf"\b({_NAMES})\s+(?:is|was)\s+(?:correct|right)\b", re.IGNORECASE),
)
# Keeping one's own answer without restating it.
_KEEP = re.compile(r"\b(?:maintain|keep|stand\s+by|stick\s+(?:with|to)|retain)\s+my\s+"
                   r"(?:original|previous|initial|earlier|first)?\s*answer\b"
                   r"|\bmy\s+(?:original|previous|initial|earlier)\s+answer\s+(?:is|was|remains)\s+(?:correct|right)",
                   re.IGNORECASE)


@dataclass
class Peer:
    model: str
    answer: Optional[str]
    response: str


def peer_order(peers: Sequence[Peer], seed: int, task: str, example_id: str, model: str,
               round_index: int) -> List[Peer]:
    """A fresh shuffle per question, agent and round, so position cannot favour any one model."""
    shuffled = list(peers)
    random.Random(f"{seed}:{task}:{example_id}:{model}:{round_index}").shuffle(shuffled)
    return shuffled


def revision_messages(task, prompt: str, own_response: str, peers: Sequence[Peer]) -> List[Dict[str, str]]:
    """The question, the agent's own previous response in its own voice, then its peers under neutral names."""
    blocks = [PEER(label=label, answer=peer.answer or "no clear answer", response=peer.response)
              for label, peer in zip(PEER_LABELS, peers)]
    request = REVISE(intro=intro(len(peers)), peers="\n\n".join(blocks),
                     word_limit=TASKS[task.name]["word_limit"], answer_format=answer_format(task))
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": own_response},
            {"role": "user", "content": request}]


def answer_span(task, text: str):
    """The task's extractor, minus cues glued to a word: "Beta's answer is correct" is not option C."""
    while text:
        found = task.extract_answer(text)
        glued = (found and found.explicit and len(found.text) == 1 and found.end < len(text)
                 and text[found.end].isalpha())
        if not glued:
            return found
        text = text[:found.start] + "#" + text[found.end:]
    return None


def resolve_answer(task, text: str, own_previous: Optional[str],
                   peers: Sequence[Peer]) -> Tuple[Optional[str], str]:
    """The answer a revision commits to, and how it was found.

    explicit: the task's answer cue ("The answer is ..."); peer_reference: it names a peer it agrees with;
    kept_previous: it keeps its own answer without restating it; inferred: the extractor's weak fallback.
    """
    found = answer_span(task, text)
    if found and found.explicit:
        return found.text, "explicit"
    labels = {label.lower(): peer for label, peer in zip(PEER_LABELS, peers)}
    named = {match.group(1).lower() for pattern in _AGREEMENT for match in pattern.finditer(text or "")}
    answers = [labels[name].answer for name in named if name in labels and labels[name].answer is not None]
    if answers and all(task.equivalent(answer, answers[0]) for answer in answers):
        return answers[0], "peer_reference"
    if own_previous is not None and _KEEP.search(text or ""):
        return own_previous, "kept_previous"
    if found:
        return found.text, "inferred"
    return None, "none"


def agreement(task, answer: Optional[str], own_previous: Optional[str],
              peers: Sequence[Peer]) -> Dict[str, object]:
    """Who the revised answer agrees with, and whether it moved to a peer's previous answer."""
    def same(a, b):
        return a is not None and b is not None and task.equivalent(a, b)

    changed = not same(answer, own_previous) if (answer or own_previous) else False
    agrees = [peer.model for peer in peers if same(answer, peer.answer)]
    return {"previous_answer": own_previous, "changed": changed, "agrees_with": agrees,
            "adopted_from": agrees if changed else []}
