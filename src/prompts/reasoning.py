"""Zero-shot reasoning prompts for open-response tasks."""

from .base import Template


class NumericReasoning:
    solve = Template('{question}\n\nSolve the problem step by step. End your response with "The answer is <number>."')
    answer_prefix = "The answer is "


class BoxedReasoning:
    solve = Template("{question}\n\nSolve the problem step by step. Put your final answer within \\boxed{{}}.")
    answer_prefix = "The final answer is \\boxed{"
