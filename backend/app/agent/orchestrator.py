from __future__ import annotations

import logging
from uuid import uuid4

from app.agent.actions import AgentResponse
from app.agent.errors import WorkflowNotFoundError
from app.agent.policy import missing_required
from app.agent.state import (
    AgentAction,
    AgentState,
    ClarificationState,
    MessageRole,
    WorkflowError,
    WorkflowMessage,
    WorkflowStatus,
)
from app.agent.store import WorkflowStore
from app.decision.budget import (
    BudgetEstimator,
    budget_input_from_requirements,
)
from app.decision.engine import (
    Candidate,
    DecisionEngine,
    Recommendation,
    RecommendationInput,
    RecommendationStatus,
)
from app.models.travel_requirements import (
    ClarificationItem,
    ExtractionResult,
    TravelRequirements,
)
from app.services.requirement_extraction import (
    ExtractionError,
    InconsistentExtractionError,
    LLMOutputValidationError,
    LLMProviderError,
    RequirementExtractionService,
)
from app.tools.search import SearchError, SearchProvider, SearchRequest

log = logging.getLogger("travel_genie.agent")


def _error_code(exc: Exception) -> str:
    if isinstance(exc, LLMOutputValidationError):
        return "llm_invalid_output"
    if isinstance(exc, InconsistentExtractionError):
        return "inconsistent_extraction"
    if isinstance(exc, LLMProviderError):
        return exc.code
    return "extraction_failed"


def _clarification_reply(required: list[ClarificationItem]) -> str:
    lines = [f"- {item.field}: {item.question}" for item in required]
    return "A few details are still missing:\n" + "\n".join(lines)


def _summarize(req: TravelRequirements) -> str:
    parts: list[str] = []
    trip = req.trip
    route = " to ".join(x for x in (trip.origin, trip.destination) if x)
    if route:
        parts.append(route)
    dates = trip.dates
    if dates is not None:
        if dates.date_range is not None:
            parts.append(
                f"{dates.date_range.start.isoformat()} \u2192 {dates.date_range.end.isoformat()}"
            )
        elif dates.duration_days is not None:
            parts.append(f"{dates.duration_days} day(s)")
    if req.travelers.count is not None:
        parts.append(f"{req.travelers.count} traveler(s)")
    budget = req.budget
    if budget.total is not None:
        parts.append(f"budget {budget.total} {budget.currency or ''}".rstrip())
    summary = "\n".join(f"- {part}" for part in parts)
    return summary or "- (no requirements extracted yet)"


class TravelAgent:
    """Deterministic orchestrator for one Travel Genie conversation.

    The agent owns workflow state and sequences extraction, clarification and
    the search handoff. All state transitions are deterministic; the only LLM
    path is ``RequirementExtractionService``.
    """

    def __init__(
        self,
        *,
        extraction: RequirementExtractionService,
        store: WorkflowStore,
        search_provider: SearchProvider | None = None,
        decision_engine: DecisionEngine | None = None,
        budget_estimator: BudgetEstimator | None = None,
    ) -> None:
        self._extraction = extraction
        self._store = store
        self._search_provider = search_provider
        self._decision_engine = decision_engine
        self._budget_estimator = budget_estimator

    def create_workflow(self) -> str:
        workflow_id = uuid4().hex
        self._store.create(AgentState(workflow_id=workflow_id))
        return workflow_id

    def get_state(self, workflow_id: str) -> AgentState:
        state = self._store.get(workflow_id)
        if state is None:
            raise WorkflowNotFoundError(workflow_id)
        return state

    def handle_message(self, workflow_id: str, text: str) -> AgentResponse:
        state = self.get_state(workflow_id)
        if state.status is WorkflowStatus.FAILED:
            return self._terminal_error_response(state)

        state.conversation.append(WorkflowMessage(role=MessageRole.USER, text=text))
        prior_user_text = [
            m.text for m in state.conversation if m.role is MessageRole.USER
        ][:-1]
        try:
            result = self._extraction.extract(text, context=prior_user_text)
        except ExtractionError as exc:
            return self._record_failure(state, exc)

        return self._continue_turn(state, result)

    def handle_structured(
        self, workflow_id: str, result: ExtractionResult
    ) -> AgentResponse:
        """Continue a workflow from an already-validated ``ExtractionResult``.

        Used by the deterministic structured-input adapter. It shares the exact
        turn logic (clarification policy, search handoff, planning) with the
        free-text path, so there is only one orchestration path.
        """
        state = self.get_state(workflow_id)
        if state.status is WorkflowStatus.FAILED:
            return self._terminal_error_response(state)

        state.conversation.append(
            WorkflowMessage(
                role=MessageRole.USER,
                text=result.metadata.original_user_message,
            )
        )
        return self._continue_turn(state, result)

    def _continue_turn(
        self, state: AgentState, result: ExtractionResult
    ) -> AgentResponse:
        state.last_extraction = result
        state.requirements = result.requirements

        required = missing_required(result)
        if required:
            response = AgentResponse(
                workflow_id=state.workflow_id,
                status=WorkflowStatus.AWAITING_CLARIFICATION,
                action=AgentAction.ASK_CLARIFICATION,
                reply=_clarification_reply(required),
                clarification_questions=required,
                requirements=result.requirements,
            )
            state.clarification = ClarificationState(pending=required)
        else:
            response = AgentResponse(
                workflow_id=state.workflow_id,
                status=WorkflowStatus.REQUIREMENTS_COMPLETE,
                action=AgentAction.SEARCH,
                reply=(
                    "Requirements are complete.\n"
                    + _summarize(result.requirements)
                    + "\nNext step: search travel options."
                ),
                requirements=result.requirements,
                search_handoff=SearchRequest(
                    workflow_id=state.workflow_id, requirements=result.requirements
                ),
            )
            state.clarification = ClarificationState()
            self._run_planning(state, response)
        return self._complete_turn(state, response)

    def _run_planning(self, state: AgentState, response: AgentResponse) -> None:
        """Run search + deterministic decision, persisting the outcome.

        Planning only runs once requirements are complete and only when a search
        provider and decision engine are wired in. Any failure is represented
        explicitly on the recommendation; it is never disguised as success and
        never produces an unhandled error.
        """
        if (
            self._search_provider is None
            or self._decision_engine is None
            or state.requirements is None
        ):
            return

        # A new planning attempt always starts from a clean estimate so a failed
        # turn can never leave a stale budget from a previous recommendation.
        state.budget_estimate = None

        request = SearchRequest(
            workflow_id=state.workflow_id, requirements=state.requirements
        )
        try:
            search_response = self._search_provider.search(request)
        except SearchError as exc:
            log.warning("Search provider unavailable: %s", exc.code)
            self._record_planning_status(
                state, response, RecommendationStatus.PROVIDER_UNAVAILABLE
            )
            return
        except Exception:  # noqa: BLE001 - never let a provider bug 500 the turn
            log.exception("Unexpected search provider failure")
            self._record_planning_status(
                state, response, RecommendationStatus.PROVIDER_UNAVAILABLE
            )
            return

        state.search_results = list(search_response.results)
        try:
            recommendation = self._decision_engine.recommend(
                RecommendationInput(
                    workflow_id=state.workflow_id,
                    requirements=state.requirements,
                    candidates=[
                        Candidate.from_search_result(result)
                        for result in state.search_results
                    ],
                )
            )
        except Exception:  # noqa: BLE001 - never let a decision bug 500 the turn
            log.exception("Decision engine failure")
            self._record_planning_status(
                state, response, RecommendationStatus.DECISION_ERROR
            )
            return

        state.recommendation = recommendation
        response.recommendation = recommendation
        self._run_budget_estimate(state, response)

    def _run_budget_estimate(
        self, state: AgentState, response: AgentResponse
    ) -> None:
        """Estimate a transparent budget from completed requirements.

        Runs whenever the decision engine returns a ``Recommendation`` (including
        ``no_results`` and ``no_feasible_options``); provider or decision
        exceptions are recorded before this step and leave the estimate absent.
        It is strictly informational: it never changes recommendation status,
        filtering, scoring, approval, or execution. When no meaningful estimate
        can be calculated the estimate is left absent rather than fabricated, and
        a failing estimator is contained so it never breaks the turn.
        """
        if self._budget_estimator is None or state.requirements is None:
            return
        try:
            estimator_input = budget_input_from_requirements(
                state.requirements, state.recommendation
            )
            if estimator_input is None:
                return
            estimate = self._budget_estimator.estimate(estimator_input)
        except Exception:  # noqa: BLE001 - never let estimation break a turn
            log.exception("Budget estimation failure")
            return
        state.budget_estimate = estimate
        response.budget_estimate = estimate

    def _record_planning_status(
        self,
        state: AgentState,
        response: AgentResponse,
        status: RecommendationStatus,
    ) -> None:
        recommendation = Recommendation(workflow_id=state.workflow_id, status=status)
        state.recommendation = recommendation
        response.recommendation = recommendation

    def _complete_turn(self, state: AgentState, response: AgentResponse) -> AgentResponse:
        state.status = response.status
        state.current_action = response.action
        state.conversation.append(
            WorkflowMessage(role=MessageRole.ASSISTANT, text=response.reply)
        )
        self._store.save(state)
        return response

    def _record_failure(self, state: AgentState, exc: ExtractionError) -> AgentResponse:
        error = WorkflowError(code=_error_code(exc), message=str(exc))
        state.status = WorkflowStatus.FAILED
        state.errors.append(error)
        response = AgentResponse(
            workflow_id=state.workflow_id,
            status=WorkflowStatus.FAILED,
            action=AgentAction.REPORT_ERROR,
            reply=(
                "I couldn't understand your requirements "
                f"({error.code}). Please start a new conversation."
            ),
            error=error,
        )
        return self._complete_turn(state, response)

    def _terminal_error_response(self, state: AgentState) -> AgentResponse:
        error = state.errors[-1]
        response = AgentResponse(
            workflow_id=state.workflow_id,
            status=WorkflowStatus.FAILED,
            action=AgentAction.AWAIT_USER,
            reply=(
                f"This conversation ended with an error "
                f"({error.code}). Please start a new conversation."
            ),
            error=error,
        )
        state.conversation.append(
            WorkflowMessage(role=MessageRole.ASSISTANT, text=response.reply)
        )
        self._store.save(state)
        return response