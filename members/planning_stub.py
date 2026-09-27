"""Fixed planning integration entry; replace preview with member planner."""

from members.planning.reference_preview import build_preview


def plan(perception, decision):
    return build_preview(perception, decision)
