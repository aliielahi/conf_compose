"""Revision prompt: each agent sees its own previous answer and every peer's previous answer and reasoning."""

from conf_compose.prompts.base import Template

# Names, not letters or numbers, so a reference like "I agree with Beta" can never be extracted as an answer.
PEER_LABELS = ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")

PEER = Template("{label}'s answer: {answer}\n{label}'s reasoning:\n{response}")

REVISE = Template(
    "{intro}\n\n{peers}\n\n"
    "Use their reasoning as evidence, but judge it critically: any of them may be wrong, and so may your own "
    "previous answer. Solve the question again step by step concisely, in at most {word_limit} words. "
    "Even if you agree with another agent or keep your previous answer, state your final answer in full "
    "rather than referring to an agent. {answer_format}"
)


def intro(count: int) -> str:
    if count == 1:
        return "Another agent answered the same question. Its answer and reasoning:"
    return f"{count} other agents answered the same question. Their answers and reasoning:"


def answer_format(task) -> str:
    """The closing instruction of the task's own solve prompt, so round 0 and later rounds ask for one format."""
    text = task.prompts.solve.text
    marker = "words. "
    if marker not in text:
        raise ValueError(f"{task.name}: cannot find the answer-format instruction in its solve prompt")
    return text[text.index(marker) + len(marker):].replace("{{", "{").replace("}}", "}")
