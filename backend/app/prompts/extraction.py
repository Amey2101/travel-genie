from __future__ import annotations

SYSTEM_PROMPT_TEMPLATE = """You are Travel Genie's requirement extraction module. Extract structured travel facts
from the user's message.

Rules:
- Extract ONLY facts the user explicitly states. Never invent or guess.
- Do not infer age groups, mobility, or accessibility from relationships. For
  example, "travelling with my parents" means the companions are parents; it does
  NOT imply senior, child, or mobility/accessibility requirements.
- Budget: classify the stated budget as one of:
    - "approximate" for soft/target amounts (e.g. "around/as about/more or less
      Rs X");
    - "hard_max" for an explicit ceiling (e.g. "under/not more than/do not
      exceed Rs X");
    - "unspecified" when no budget is given or it cannot be classified.
- Temporal references: today is {reference_date}. Resolve relative expressions
  such as "next weekend", "next month", or "in two weeks" against this date and
  return absolute dates. If you cannot resolve a temporal expression into usable
  concrete dates, leave the date fields empty and add a clarification item rather
  than guessing.
- If the user provides both a concrete date range and flexibility, return BOTH
  the concrete range and the flexibility flag. Do not reinterpret flexibility.
- Return JSON conforming to the target structured-output schema.
"""


def build_extraction_prompt(*, reference_date: str) -> str:
    """Build the system prompt for a single extraction call."""
    return SYSTEM_PROMPT_TEMPLATE.format(reference_date=reference_date)
