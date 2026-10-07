"""Bounded local-generation contracts and services."""

from academic_chatbot.generation.context import (
    MVP_CITED_ANSWER_JSON_SCHEMA,
    MVP_SYSTEM_MESSAGE,
    RenderedGenerationContext,
    render_generation_context,
)
from academic_chatbot.generation.models import (
    MVP_GENERATION_PROFILE_ID,
    MVP_MODEL_PROFILE_ID,
    MVP_OUTPUT_SCHEMA_ID,
    MVP_PROMPT_PROFILE_ID,
    MVP_RESULT_SCHEMA_ID,
    AnswerCitation,
    LocalGenerationAbstained,
    LocalGenerationAbstentionReason,
    LocalGenerationAnswered,
    LocalGenerationFailed,
    LocalGenerationFailureCode,
    LocalGenerationRejected,
    LocalGenerationRejectionCode,
    LocalGenerationRequest,
    LocalGenerationResult,
    MvpCitedAnswer,
)
from academic_chatbot.generation.service import LocalGenerationService

__all__ = [
    "MVP_CITED_ANSWER_JSON_SCHEMA",
    "MVP_GENERATION_PROFILE_ID",
    "MVP_MODEL_PROFILE_ID",
    "MVP_OUTPUT_SCHEMA_ID",
    "MVP_PROMPT_PROFILE_ID",
    "MVP_RESULT_SCHEMA_ID",
    "MVP_SYSTEM_MESSAGE",
    "AnswerCitation",
    "LocalGenerationAbstained",
    "LocalGenerationAbstentionReason",
    "LocalGenerationAnswered",
    "LocalGenerationFailed",
    "LocalGenerationFailureCode",
    "LocalGenerationRejected",
    "LocalGenerationRejectionCode",
    "LocalGenerationRequest",
    "LocalGenerationResult",
    "LocalGenerationService",
    "MvpCitedAnswer",
    "RenderedGenerationContext",
    "render_generation_context",
]
