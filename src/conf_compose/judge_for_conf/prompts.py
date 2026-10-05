"""Prompts for the judge baseline: three levels of what the judge is shown about each panel member."""

from conf_compose.prompts.base import Template

# How much of each panel member is revealed to the judge.
LEVELS = ("answer", "answer_reasoning", "answer_reasoning_confidence")

HEADER = Template(
    "Question: {question}\n\n{members}\n\n"
    "Decide the correct final answer. Think step by step concisely, in at most {word_limit} words.\n"
    'End your response with "{answer_prefix}<answer>" and then, on a new line, '
    '"Confidence: <0-10>" where 0 means guessing and 10 means almost certain.'
)

MEMBER = {
    "answer": Template("{name} answered: {answer}"),
    "answer_reasoning": Template("{name} answered: {answer}\n{name}'s reasoning:\n{reasoning}"),
    "answer_reasoning_confidence": Template(
        "{name} answered: {answer}\n{name}'s stated confidence in that answer: {confidence}\n"
        "{name}'s reasoning:\n{reasoning}"
    ),
}
