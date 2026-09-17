"""Zero-shot reasoning prompts for open-response, multiple-choice and boolean tasks."""

from .base import Template


class NumericReasoning:
    solve = Template('{question}\n\nSolve the problem step by step concisely, in at most {word_limit} words. '
                     'End your response with "The answer is <number>."')
    answer_prefix = "The answer is "


class BoxedReasoning:
    solve = Template("{question}\n\nSolve the problem step by step concisely, in at most {word_limit} words. "
                     "Put your final answer within \\boxed{{}}.")
    answer_prefix = "The final answer is \\boxed{"


class MultipleChoiceReasoning:
    solve = Template('{question}\n\nThink step by step concisely, in at most {word_limit} words. '
                     'End your response with "The answer is <letter>."')
    answer_prefix = "The answer is "


class BooleanReasoning:
    solve = Template('{question}\n\nThink step by step concisely, in at most {word_limit} words. '
                     'End your response with "The answer is true." or "The answer is false."')
    answer_prefix = "The answer is "
