"""Offline confidence composition: evidence, candidates, panels, grids, methods, evaluation and reporting."""

from .candidates import (OTHER, Support, candidate_set, label_space, state_space, support, support_of)
from .evaluate import Row, best_by_validation, evaluate, fit_intercepts, format_table
from .evidence import Item, Stream, from_debate_traces, from_zero_shot
from .grid import (FAMILIES, RULES, GridConfig, anchor_prior, attach_target_scores, build_panels, collect,
                   full_panel, group_runs, hetero_subsets, homo_subsets, model_key, rows_from, short_name,
                   shared_targets, split_runs, target_of, voter_rows)
from .methods import (Prediction, fixed_answer_methods, majority_answer, pool_methods, prior_corrected,
                      selection_methods)
from .panels import Panel, Source, available, base_model, blocks, model_panels, sources
from .report import arm, family_of, format_cell, load_runs, matched_budget, pick, size_curve

__all__ = ["OTHER", "Support", "candidate_set", "label_space", "state_space", "support", "support_of",
           "Row", "best_by_validation", "evaluate", "fit_intercepts", "format_table", "Item", "Stream",
           "from_debate_traces", "from_zero_shot", "FAMILIES", "RULES", "GridConfig", "anchor_prior",
           "attach_target_scores", "build_panels", "collect", "full_panel", "group_runs", "hetero_subsets",
           "homo_subsets", "model_key", "rows_from", "short_name", "shared_targets", "split_runs",
           "target_of", "voter_rows", "Prediction", "fixed_answer_methods", "majority_answer",
           "pool_methods", "prior_corrected", "selection_methods", "Panel", "Source", "available",
           "base_model", "blocks", "model_panels", "sources", "arm", "family_of", "format_cell",
           "load_runs", "matched_budget", "pick", "size_curve"]
