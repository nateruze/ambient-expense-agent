# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any, Optional
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.models import Gemini
from google.adk.workflow import Edge, FunctionNode, Workflow
from google.genai import types

MODEL = "gemini-3.6-flash"


class ExpenseReport(BaseModel):
    amount: float = Field(..., description="The amount of the expense in USD")
    submitter: str = Field(..., description="Name or ID of the submitter")
    category: str = Field(
        ..., description="Category of the expense (e.g., travel, meals, equipment)"
    )
    description: str = Field(..., description="Detailed description of the expense")
    date: str = Field(..., description="Date of expense submission (YYYY-MM-DD)")


class RiskAnalysis(BaseModel):
    risk_score: str = Field(..., description="Risk level: LOW, MEDIUM, or HIGH")
    policy_violations: list[str] = Field(
        default_factory=list, description="Any potential policy violations identified"
    )
    summary: str = Field(..., description="Detailed reasoning for the risk judgment")


class ExpenseDecision(BaseModel):
    status: str = Field(
        ..., description="Decision status: APPROVED, REJECTED, or AUTO_APPROVED"
    )
    decision_by: str = Field(
        ..., description="Who made the decision: auto_rule or human"
    )
    risk_analysis: Optional[dict[str, Any]] = Field(
        default=None, description="Risk analysis results if reviewed by LLM"
    )
    expense: dict[str, Any] = Field(
        ..., description="The original expense report details"
    )
    notes: Optional[str] = Field(
        default=None, description="Additional decision notes or human reviewer feedback"
    )


def _parse_expense(data: Any) -> ExpenseReport:
    """Helper to deserialize ExpenseReport from dict, Pydantic model, JSON string, or types.Content."""
    if isinstance(data, ExpenseReport):
        return data
    if isinstance(data, dict):
        return ExpenseReport(**data)
    if isinstance(data, types.Content):
        text = "".join(part.text or "" for part in data.parts if hasattr(part, "text"))
        return ExpenseReport(**json.loads(text))
    if isinstance(data, str):
        return ExpenseReport(**json.loads(data))
    raise ValueError(f"Unsupported expense report input type: {type(data)}")


def route_expense(node_input: Any) -> Event:
    """Pure Python routing function to enforce $100 threshold before any LLM is called."""
    expense = _parse_expense(node_input)
    expense_dict = expense.model_dump()
    text = json.dumps(expense_dict)
    content = types.Content(role="model", parts=[types.Part.from_text(text=text)])
    if expense.amount < 100.0:
        return Event(
            output=expense_dict,
            content=content,
            actions=EventActions(
                route="auto_approve", state_delta={"expense": expense_dict}
            ),
        )
    return Event(
        output=expense_dict,
        content=content,
        actions=EventActions(route="llm_review", state_delta={"expense": expense_dict}),
    )


def auto_approve_node(ctx: Context, node_input: dict[str, Any]) -> Event:
    """Instant auto-approval function for expenses under $100 without LLM involvement."""
    expense = ctx.state.get("expense", node_input)
    decision = ExpenseDecision(
        status="APPROVED",
        decision_by="auto_rule",
        risk_analysis={
            "risk_score": "LOW",
            "policy_violations": [],
            "summary": "Auto-approved: Expense amount is under the $100 threshold.",
        },
        expense=expense,
        notes="Auto-approved instantly under $100 policy rule.",
    )
    d = decision.model_dump()
    content = types.Content(
        role="model", parts=[types.Part.from_text(text=json.dumps(d))]
    )
    return Event(output=d, content=content)


llm_reviewer = LlmAgent(
    name="llm_reviewer",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an expert financial compliance and risk reviewer. Analyze the provided expense report details "
        "and perform a risk evaluation. Evaluate whether the expense amount, category, description, and submitter "
        "present any financial risk or policy concerns. Return a structured risk analysis with risk_score (LOW, MEDIUM, or HIGH), "
        "any potential policy_violations, and a clear summary."
    ),
    output_schema=RiskAnalysis,
    output_key="risk_analysis",
)


async def human_reviewer(ctx: Context, node_input: dict[str, Any]):
    """Human-in-the-loop function to request manual approval for expenses >= $100."""
    expense = ctx.state.get("expense", {})
    amount = expense.get("amount", "unknown")
    submitter = expense.get("submitter", "unknown")
    category = expense.get("category", "unknown")

    if not ctx.resume_inputs or "human_approval" not in ctx.resume_inputs:
        msg = (
            f"Expense approval required for ${amount} ({category}) submitted by {submitter}. "
            f"LLM Risk Analysis: {node_input.get('summary', 'No summary available')} "
            f"(Risk Score: {node_input.get('risk_score', 'UNKNOWN')}). "
            f"Please respond with 'approve' or 'reject'."
        )
        yield Event(
            output={"status": "APPROVAL_REQUIRED", "message": msg},
            content=types.Content(role="model", parts=[types.Part.from_text(text=msg)]),
        )
        return

    human_response = str(ctx.resume_inputs.get("human_approval", "")).strip().lower()
    is_approved = human_response in ("approve", "approved", "yes", "true", "1")
    status = "APPROVED" if is_approved else "REJECTED"

    decision = ExpenseDecision(
        status=status,
        decision_by="human",
        risk_analysis=node_input,
        expense=expense,
        notes=f"Human reviewer decision: {ctx.resume_inputs.get('human_approval')}",
    )
    d = decision.model_dump()
    content = types.Content(
        role="model", parts=[types.Part.from_text(text=json.dumps(d))]
    )
    yield Event(output=d, content=content)


# Explicit FunctionNodes for workflow topology
route_expense_node = FunctionNode(func=route_expense, name="route_expense")
auto_approve_node_wrapper = FunctionNode(
    func=auto_approve_node, name="auto_approve_node"
)
human_reviewer_node = FunctionNode(
    func=human_reviewer, name="human_reviewer", rerun_on_resume=True
)

root_agent = Workflow(
    name="ambient_expense_agent",
    edges=[
        ("START", route_expense_node),
        Edge(
            from_node=route_expense_node,
            to_node=auto_approve_node_wrapper,
            route="auto_approve",
        ),
        Edge(from_node=route_expense_node, to_node=llm_reviewer, route="llm_review"),
        Edge(from_node=llm_reviewer, to_node=human_reviewer_node),
    ],
    input_schema=ExpenseReport,
)

app = App(
    root_agent=root_agent,
    name="app",
)
