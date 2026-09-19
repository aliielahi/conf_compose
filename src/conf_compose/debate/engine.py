"""Synchronous debate: round 0 is independent, later rounds show each agent its peers' previous answers.

Traces keep the exact messages every agent saw, so confidence can be estimated later on the real context.
Confidence is private unless `share_confidence` is set. Anything that changes what is sent in a later round
(for example a gate on peer messages) changes generation and needs a new execution id, not offline reweighting.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from conf_compose.prompts import DebatePrompts

from .protocol import DebateConfig
from .trace import ExampleTrace, Turn, append_traces, completed_ids, turn_id


def run_debate(llms: Sequence, task, examples, config: Optional[DebateConfig] = None,
               out_path: Optional[Path] = None, batch_size: int = 0) -> List[ExampleTrace]:
    """Run every example through all rounds, appending finished traces and skipping ones already saved."""
    config = config or DebateConfig()
    if len(llms) != config.agents:
        raise ValueError(f"{len(llms)} models given for {config.agents} agents")
    done = completed_ids(out_path) if out_path else set()
    pending = [example for example in examples if example.id not in done]
    print(f"debate: {len(pending)} examples pending ({len(examples) - len(pending)} already traced)", flush=True)

    traces: List[ExampleTrace] = []
    for batch in _batches(pending, batch_size or len(pending) or 1):
        batch_traces = [ExampleTrace(example.id, example.question, example.answer) for example in batch]
        for round_index in range(config.rounds + 1):
            for agent, llm in enumerate(llms):
                _run_round(llm, task, batch, batch_traces, agent, round_index, config)
        if out_path:
            append_traces(out_path, batch_traces)
        traces += batch_traces
    return traces


def _run_round(llm, task, examples, traces: List[ExampleTrace], agent: int, round_index: int,
               config: DebateConfig) -> None:
    conversations = [_conversation(task, example, trace, agent, round_index, config)
                     for example, trace in zip(examples, traces)]
    start = time.time()
    generations = llm.generate([messages for messages, _ in conversations], **config.generation_settings())
    seconds = (time.time() - start) / max(len(generations), 1)

    for trace, (messages, parents), generation in zip(traces, conversations, generations):
        answer = task.extract_answer(generation.text) if generation.text else None
        trace.turns.append(Turn(
            example_id=trace.example_id,
            agent=agent,
            model=llm.model,
            round=round_index,
            parents=parents,
            messages=messages,
            response=generation.text,
            answer=answer.text if answer else None,
            explicit_answer=bool(answer and answer.explicit),
            finish_reason=generation.finish_reason,
            logprobs=generation.logprobs,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            seconds=seconds,
            execution=config.execution,
            settings=config.generation_settings(),
            error=generation.error,
        ))


def _conversation(task, example, trace: ExampleTrace, agent: int, round_index: int, config: DebateConfig):
    """Messages the agent sees this round, plus the ids of the turns those messages come from."""
    question = task.prompt(example)
    if round_index == 0:
        return [{"role": "user", "content": question}], []

    own = _turn(trace, agent, round_index - 1)
    peers = [turn for turn in trace.round_turns(round_index - 1) if turn.agent != agent]
    messages = [{"role": "user", "content": question},
                {"role": "assistant", "content": own.response},
                {"role": "user", "content": _revision_request(peers, config)}]
    return messages, [turn.id for turn in peers] + [own.id]


def _revision_request(peers: Sequence[Turn], config: DebateConfig) -> str:
    blocks = []
    for peer in peers:
        fields = {"agent": peer.agent, "answer": peer.answer or "no answer", "response": peer.response}
        if config.share_confidence:
            confidence = peer.confidence.get("shared")
            blocks.append(DebatePrompts.peer_with_confidence(
                confidence="unknown" if confidence is None else f"{confidence:.2f}", **fields))
        else:
            blocks.append(DebatePrompts.peer(**fields))
    return DebatePrompts.revise(peers="\n\n".join(blocks), word_limit=config.word_limit)


def _turn(trace: ExampleTrace, agent: int, round_index: int) -> Turn:
    wanted = turn_id(trace.example_id, round_index, agent)
    return next(turn for turn in trace.turns if turn.id == wanted)


def _batches(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]
