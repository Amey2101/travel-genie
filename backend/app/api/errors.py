"""Consistent, secret-safe HTTP error handling.

Ported and adapted from the teammate implementation's ``app.py`` exception
handlers. Every API error uses a single envelope::

    {"error": {"code": "...", "message": "...", "details": [...]}}

The generic handler never echoes exception details, so internal errors and
secret material cannot leak to clients.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("travel_genie.app")

FIELD_LABELS = {
    "origin": "Starting city",
    "destination": "Destination",
    "departure_date": "Departure date",
    "return_date": "Return date",
    "duration_days": "Trip duration",
    "travelers": "Number of travelers",
    "budget": "Target budget",
    "currency": "Currency",
    "travel_style": "Travel style",
    "interests": "Interests",
    "accommodation": "Accommodation preference",
    "notes": "Additional preferences",
    "text": "Message",
}


def api_error(
    status: int,
    code: str,
    message: str,
    details: list | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": details or []}},
    )


def _friendly(err: dict) -> str:
    """Turn a Pydantic error entry into a short, human-readable message."""
    kind = err.get("type", "")
    ctx = err.get("ctx") or {}
    msg = err.get("msg", "Invalid value")
    if kind == "missing":
        return "is required."
    if kind == "string_too_short":
        return "is required." if ctx.get("min_length", 0) <= 2 else "is too short."
    if kind == "string_too_long":
        return f"must be at most {ctx.get('max_length')} characters."
    if kind in ("date_parsing", "date_from_datetime_parsing", "date_type"):
        return "must be a valid date (YYYY-MM-DD)."
    if kind in ("int_parsing", "float_parsing", "int_from_float"):
        return "must be a whole number." if "int" in kind else "must be a number."
    if kind == "greater_than":
        return f"must be greater than {ctx.get('gt'):g}."
    if kind == "greater_than_equal":
        return f"must be at least {ctx.get('ge'):g}."
    if kind == "less_than_equal":
        return f"must be at most {ctx.get('le'):g}."
    if kind == "literal_error":
        return "has an unsupported value."
    return msg.removeprefix("Value error, ")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details = []
        for err in exc.errors():
            loc = [str(p) for p in err.get("loc", []) if p != "body"]
            field = loc[0] if loc else ""
            label = FIELD_LABELS.get(field, "")
            text = _friendly(err)
            message = (
                f"{label} {text}"
                if label and text[:1].islower()
                else (f"{label}: {text}" if label else text)
            )
            details.append({"field": field, "message": message})
        summary = (
            "Please fix the highlighted details: "
            + " ".join(d["message"] for d in details)
            if details
            else "Invalid request."
        )
        return api_error(422, "validation_error", summary, details)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return api_error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error")
        return api_error(
            500,
            "internal_error",
            "Something went wrong while processing your request. Please try again.",
        )
