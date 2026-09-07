from __future__ import annotations

from decimal import Decimal

from app.providers.llm import BudgetExtraction, BudgetKind, LLMExtractionOutput


def _budget_output(kind: BudgetKind, amount: Decimal | None = Decimal("30000")) -> LLMExtractionOutput:
    return LLMExtractionOutput(
        trip_destination="Goa",
        traveler_count=2,
        budget=BudgetExtraction(amount=amount, currency="INR", kind=kind),
    )


class TestBudgetSemantics:
    def test_approximate_budget_no_hard_constraint(
        self, make_service
    ) -> None:
        output = _budget_output(BudgetKind.APPROXIMATE)
        result = make_service(output).extract("Goa trip")
        req = result.requirements

        assert req.budget.total == Decimal("30000")
        assert req.budget.currency == "INR"
        assert req.budget.is_approximate is True
        assert req.hard_constraints.max_budget is None

    def test_hard_max_budget_sets_hard_constraint(self, make_service) -> None:
        output = _budget_output(BudgetKind.HARD_MAX)
        result = make_service(output).extract("do not exceed 30k")
        req = result.requirements

        assert req.budget.total == Decimal("30000")
        assert req.budget.is_approximate is False
        assert req.hard_constraints.max_budget == Decimal("30000")

    def test_unspecified_budget_no_numbers(
        self, make_service
    ) -> None:
        output = _budget_output(BudgetKind.UNSPECIFIED, amount=None)
        result = make_service(output).extract("no budget")
        req = result.requirements

        assert req.budget.total is None
        assert req.budget.is_approximate is False
        assert req.hard_constraints.max_budget is None

    def test_hard_max_without_amount_sets_no_constraint(
        self, make_service
    ) -> None:
        output = _budget_output(BudgetKind.HARD_MAX, amount=None)
        req = make_service(output).extract("keep it cheap").requirements
        assert req.budget.total is None
        assert req.hard_constraints.max_budget is None

    def test_no_per_category_populated(
        self, make_service
    ) -> None:
        output = _budget_output(BudgetKind.HARD_MAX)
        req = make_service(output).extract("do not exceed 30k").requirements
        assert req.budget.per_category == {}


class TestBudgetPipeline:
    def test_approximate_message_semantics(self, make_service) -> None:
        """Around Rs 30k -> soft budget, never a hard ceiling."""
        result = make_service(_budget_output(BudgetKind.APPROXIMATE)).extract(
            "I want a trip around Rs 30,000"
        )
        req = result.requirements
        assert req.budget.is_approximate is True
        assert req.hard_constraints.max_budget is None

    def test_hard_message_semantics(self, make_service) -> None:
        """Do not exceed Rs 30k -> both budget detail and hard ceiling."""
        result = make_service(_budget_output(BudgetKind.HARD_MAX)).extract(
            "do not exceed Rs 30,000"
        )
        req = result.requirements
        assert req.hard_constraints.max_budget == Decimal("30000")
        assert req.budget.is_approximate is False