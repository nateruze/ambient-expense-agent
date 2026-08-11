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

import pytest
from google.adk.runners import InMemoryRunner
from google.genai import types

from app.agent import (
    ExpenseReport,
    _parse_expense,
    auto_approve_node,
    human_reviewer,
    route_expense,
)
from app.agent import (
    app as adk_app,
)


def test_expense_report_schema() -> None:
    """Test parsing valid ExpenseReport objects."""
    report = ExpenseReport(
        amount=45.50,
        submitter="Alice",
        category="meals",
        description="Team lunch",
        date="2026-08-11",
    )
    assert report.amount == 45.50
    assert report.submitter == "Alice"


def test_parse_expense_helper() -> None:
    """Test _parse_expense helper across dict, JSON string, Content, and ExpenseReport inputs."""
    data_dict = {
        "amount": 99.0,
        "submitter": "Alice",
        "category": "meals",
        "description": "Lunch",
        "date": "2026-08-11",
    }
    # From dict
    parsed_dict = _parse_expense(data_dict)
    assert parsed_dict.amount == 99.0

    # From string
    parsed_str = _parse_expense(json.dumps(data_dict))
    assert parsed_str.submitter == "Alice"

    # From Content
    content = types.Content(
        role="user", parts=[types.Part.from_text(text=json.dumps(data_dict))]
    )
    parsed_content = _parse_expense(content)
    assert parsed_content.category == "meals"

    # From ExpenseReport object
    parsed_obj = _parse_expense(parsed_dict)
    assert parsed_obj is parsed_dict


def test_route_expense_under_100() -> None:
    """Test pure Python routing for expense under $100 -> auto_approve."""
    report = ExpenseReport(
        amount=99.99,
        submitter="Bob",
        category="supplies",
        description="Notebooks",
        date="2026-08-11",
    )
    event = route_expense(report)
    assert event.actions.route == "auto_approve"
    assert event.output["amount"] == 99.99
    assert event.actions.state_delta["expense"]["amount"] == 99.99


def test_route_expense_100_or_more() -> None:
    """Test pure Python routing for expense >= $100 -> llm_review."""
    report = ExpenseReport(
        amount=100.00,
        submitter="Charlie",
        category="travel",
        description="Flight ticket",
        date="2026-08-11",
    )
    event = route_expense(report)
    assert event.actions.route == "llm_review"
    assert event.output["amount"] == 100.00


class DummyContext:
    def __init__(self, state=None, resume_inputs=None):
        self.state = state or {}
        self.resume_inputs = resume_inputs or {}


def test_auto_approve_node() -> None:
    """Test auto_approve_node returns instant approval without LLM."""
    expense_data = {
        "amount": 50.0,
        "submitter": "Dave",
        "category": "software",
        "description": "IDE License",
        "date": "2026-08-11",
    }
    ctx = DummyContext(state={"expense": expense_data})
    event = auto_approve_node(ctx, expense_data)
    output = event.output
    assert output["status"] == "APPROVED"
    assert output["decision_by"] == "auto_rule"
    assert output["expense"]["amount"] == 50.0


@pytest.mark.asyncio
async def test_human_reviewer_interrupt() -> None:
    """Test human_reviewer yields Event and RequestInput when no resume_inputs provided."""
    ctx = DummyContext(
        state={"expense": {"amount": 250.0, "submitter": "Eve", "category": "travel"}},
        resume_inputs={},
    )
    risk_info = {"risk_score": "MEDIUM", "summary": "Unusually high flight cost."}

    gen = human_reviewer(ctx, risk_info)
    first_event = await anext(gen)
    second_event = await anext(gen)
    assert first_event.output["status"] == "APPROVAL_REQUIRED"
    assert second_event.interrupt_id == "human_approval"
    assert "250.0" in second_event.message


@pytest.mark.asyncio
async def test_human_reviewer_approve() -> None:
    """Test human_reviewer produces APPROVED decision when resume_inputs contains approval."""
    ctx = DummyContext(
        state={"expense": {"amount": 250.0, "submitter": "Eve", "category": "travel"}},
        resume_inputs={"human_approval": "approve"},
    )
    risk_info = {"risk_score": "MEDIUM", "summary": "Unusually high flight cost."}

    gen = human_reviewer(ctx, risk_info)
    event = await anext(gen)
    output = event.output
    assert output["status"] == "APPROVED"
    assert output["decision_by"] == "human"


@pytest.mark.asyncio
async def test_human_reviewer_reject() -> None:
    """Test human_reviewer produces REJECTED decision when resume_inputs contains rejection."""
    ctx = DummyContext(
        state={
            "expense": {"amount": 500.0, "submitter": "Frank", "category": "equipment"}
        },
        resume_inputs={"human_approval": "reject"},
    )
    risk_info = {"risk_score": "HIGH", "summary": "Exceeds equipment cap."}

    gen = human_reviewer(ctx, risk_info)
    event = await anext(gen)
    output = event.output
    assert output["status"] == "REJECTED"
    assert output["decision_by"] == "human"


@pytest.mark.asyncio
async def test_workflow_auto_approve_execution() -> None:
    """Integration style test for sub-$100 workflow execution with InMemoryRunner (no LLM required)."""
    runner = InMemoryRunner(app=adk_app)
    session = await runner.session_service.create_session(
        app_name="app", user_id="test_user"
    )

    payload = json.dumps(
        {
            "amount": 75.0,
            "submitter": "Grace",
            "category": "meals",
            "description": "Client dinner",
            "date": "2026-08-11",
        }
    )

    outputs = []
    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part.from_text(text=payload)]
        ),
    ):
        if event.output is not None:
            outputs.append(event.output)

    assert len(outputs) > 0
    final_output = outputs[-1]
    assert final_output["status"] == "APPROVED"
    assert final_output["decision_by"] == "auto_rule"
