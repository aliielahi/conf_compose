"""Prompts for the judge as a confidence combiner: the answer is given, the judge returns a confidence."""

from conf_compose.prompts.base import Template

# What the judge is shown about each panel member. The aggregated answer is always given.
VIEWS = ("reasoning", "reasoning_confidence")

HEADER = Template(
    "Question: {question}\n\n{members}\n\n"
    "The system's final answer, decided by majority vote over those models, is: {final_answer}\n\n"
    "{instruction} Think step by step concisely, in at most {word_limit} words.\n"
    'Then write "Justification: <one sentence>" giving the single main reason for your confidence, '
    'and on a new line "Confidence: <0-10>", where 0 means the final answer is almost certainly '
    "wrong and 10 means it is almost certainly correct."
)

INSTRUCTION = {
    "reasoning": "Judge how likely that final answer is to be correct, from the quality and the "
                 "soundness of the reasoning above.",
    "reasoning_confidence": "Each model also reported how confident it was. Combine those confidences "
                            "with the quality of the reasoning to judge how likely the final answer is "
                            "to be correct.",
}

MEMBER = {
    "reasoning": Template("{name} answered: {answer}\n{name}'s reasoning:\n{reasoning}"),
    "reasoning_confidence": Template(
        "{name} answered: {answer}\n{name}'s confidence in its own answer ({method}): {confidence}\n"
        "{name}'s reasoning:\n{reasoning}"
    ),
}

# A plain context for scoring candidate answers; the answer prefix is passed to score() separately.
SCORING = Template("Question: {question}\n\n{members}\n\nWhat is the correct answer?")
