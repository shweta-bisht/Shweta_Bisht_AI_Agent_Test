"""Conditional routing logic for the CommerceAI LangGraph pipeline."""

from __future__ import annotations

from typing import List

from workflow.state import CommerceAIState


def route_after_cleaning(state: CommerceAIState) -> List[str] | str:
    """Route to execution summary on fatal ingestion/cleaning failure, else fan out."""
    if state.get("fatal_error"):
        return "execution_summary"
    return ["content_generation", "sales_analysis", "competitor_research"]
