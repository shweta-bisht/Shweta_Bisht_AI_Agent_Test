"""LangGraph StateGraph construction and execution entrypoint."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.competitor_agent import CompetitorAnalysisAgent
from agents.content_agent import ContentGenerationAgent
from agents.image_agent import LifestyleImageAgent
from agents.sales_agent import SalesAnalysisAgent
from config import AppConfig
from utils.cost_tracker import CostTracker
from workflow.nodes import WorkflowNodes
from workflow.routing import route_after_cleaning
from workflow.state import CommerceAIState


def build_commerce_graph(
    config: AppConfig,
    cost_tracker: Optional[CostTracker] = None,
    content_agent: Optional[ContentGenerationAgent] = None,
    image_agent: Optional[LifestyleImageAgent] = None,
    competitor_agent: Optional[CompetitorAnalysisAgent] = None,
    sales_agent: Optional[SalesAnalysisAgent] = None,
):
    """Compile the CommerceAI LangGraph workflow with checkpointing and parallel branches."""
    nodes = WorkflowNodes(
        config=config,
        cost_tracker=cost_tracker,
        content_agent=content_agent,
        image_agent=image_agent,
        competitor_agent=competitor_agent,
        sales_agent=sales_agent,
    )

    builder = StateGraph(CommerceAIState)
    builder.add_node("excel_ingestion", nodes.excel_ingestion)
    builder.add_node("data_cleaning", nodes.data_cleaning)
    builder.add_node("content_generation", nodes.content_generation)
    builder.add_node("image_generation", nodes.image_generation)
    builder.add_node("sales_analysis", nodes.sales_analysis)
    builder.add_node("competitor_research", nodes.competitor_research)
    builder.add_node("output_generation", nodes.output_generation)
    builder.add_node("execution_summary", nodes.execution_summary)

    builder.add_edge(START, "excel_ingestion")
    builder.add_edge("excel_ingestion", "data_cleaning")

    builder.add_conditional_edges(
        "data_cleaning",
        route_after_cleaning,
        {
            "execution_summary": "execution_summary",
            "content_generation": "content_generation",
            "sales_analysis": "sales_analysis",
            "competitor_research": "competitor_research",
        },
    )

    builder.add_edge("content_generation", "image_generation")
    builder.add_edge(
        ["image_generation", "sales_analysis", "competitor_research"],
        "output_generation",
    )
    builder.add_edge("output_generation", "execution_summary")
    builder.add_edge("execution_summary", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def run_commerce_workflow(
    config: AppConfig,
    run_id: Optional[str] = None,
    cost_tracker: Optional[CostTracker] = None,
    content_agent: Optional[ContentGenerationAgent] = None,
    image_agent: Optional[LifestyleImageAgent] = None,
    competitor_agent: Optional[CompetitorAnalysisAgent] = None,
    sales_agent: Optional[SalesAnalysisAgent] = None,
) -> Dict[str, Any]:
    """Execute the full compiled CommerceAI LangGraph workflow."""
    rid = run_id or uuid.uuid4().hex[:12]
    graph = build_commerce_graph(
        config=config,
        cost_tracker=cost_tracker,
        content_agent=content_agent,
        image_agent=image_agent,
        competitor_agent=competitor_agent,
        sales_agent=sales_agent,
    )
    initial_state: CommerceAIState = {
        "run_id": rid,
        "start_time_iso": datetime.now(timezone.utc).isoformat(),
        "input_workbook_path": str(config.input_workbook_path if hasattr(config, "input_workbook_path") else config.input_file),
        "errors": [],
        "warnings": [],
        "generated_files": [],
        "fatal_error": False,
    }
    return graph.invoke(initial_state, config={"configurable": {"thread_id": rid}})
