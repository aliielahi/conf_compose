"""All prompt text used in the project, grouped by purpose."""

from .base import Template
from .debate import DebatePrompts
from .confidence import ContentFreeInputs, VerbalizedPrompts, VerificationPrompts
from .reasoning import BooleanReasoning, BoxedReasoning, MultipleChoiceReasoning, NumericReasoning

__all__ = ["Template", "NumericReasoning", "BoxedReasoning", "MultipleChoiceReasoning", "BooleanReasoning",
           "DebatePrompts",
           "VerbalizedPrompts", "VerificationPrompts", "ContentFreeInputs"]
