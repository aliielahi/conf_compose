"""Debate prompts: how peer answers are shown and how an agent is asked to revise."""

from .base import Template


class DebatePrompts:
    peer = Template("Agent {agent} answered: {answer}\nIts reasoning: {response}")
    peer_with_confidence = Template("Agent {agent} answered: {answer} (confidence {confidence})\n"
                                    "Its reasoning: {response}")
    revise = Template("Other agents answered the same question:\n\n{peers}\n\n"
                      "Consider their reasoning critically, then give your own updated answer to the original "
                      "question in the same format as before, in at most {word_limit} words.")
