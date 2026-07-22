"""Stable local-model boundary shared by generation workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final, Literal, Protocol, Self, cast, runtime_checkable

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    field_validator,
    model_validator,
)

MAX_JSON_CONTAINER_DEPTH: Final = 64


def _require_nonblank_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonblank string")
    return value


def _clone_json_value(
    value: object,
    *,
    field_name: str,
    container_depth: int,
    active_container_ids: set[int],
) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite JSON numbers")
        return value
    if isinstance(value, (list, Mapping)):
        if container_depth > MAX_JSON_CONTAINER_DEPTH:
            raise ValueError(
                f"{field_name} exceeds the maximum container depth of {MAX_JSON_CONTAINER_DEPTH}"
            )
        container_id = id(value)
        if container_id in active_container_ids:
            raise ValueError(f"{field_name} must not contain cycles")
        active_container_ids.add(container_id)
        try:
            if isinstance(value, list):
                return [
                    _clone_json_value(
                        item,
                        field_name=field_name,
                        container_depth=container_depth + 1,
                        active_container_ids=active_container_ids,
                    )
                    for item in value
                ]
            cloned: dict[str, JsonValue] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{field_name} must contain only string object keys")
                cloned[key] = _clone_json_value(
                    item,
                    field_name=field_name,
                    container_depth=container_depth + 1,
                    active_container_ids=active_container_ids,
                )
            return cloned
        finally:
            active_container_ids.remove(container_id)
    raise ValueError(f"{field_name} must contain only JSON values")


def _clone_json_object(value: object, *, field_name: str) -> dict[str, JsonValue]:
    cloned = _clone_json_value(
        value,
        field_name=field_name,
        container_depth=1,
        active_container_ids=set(),
    )
    if not isinstance(cloned, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return cloned


def _freeze_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return cast(JsonValue, tuple(_freeze_json_value(item) for item in value))
    if isinstance(value, Mapping):
        frozen = {key: _freeze_json_value(item) for key, item in value.items()}
        return cast(JsonValue, MappingProxyType(frozen))
    return value


def _freeze_json_object(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return MappingProxyType({key: _freeze_json_value(item) for key, item in value.items()})


def _thaw_json_value(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        thawed: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Frozen JSON objects must contain only string keys")
            thawed[key] = _thaw_json_value(item)
        return thawed
    if isinstance(value, (tuple, list)):
        return [_thaw_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Frozen JSON state contains a non-JSON value")


def _thaw_json_object(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    thawed = _thaw_json_value(value)
    if not isinstance(thawed, dict):
        raise TypeError("Frozen JSON object did not thaw to a dictionary")
    return thawed


_FrozenJsonObject = Annotated[
    Mapping[str, JsonValue],
    AfterValidator(_freeze_json_object),
    PlainSerializer(
        _thaw_json_object,
        return_type=dict[str, JsonValue],
        when_used="always",
    ),
]


def _require_finite_nonnegative_float(value: object, *, field_name: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field_name} must be a finite nonnegative float")
    return value


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ModelMessage(_StrictFrozenModel):
    """One immutable message accepted by the local model port."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content", mode="before")
    @classmethod
    def _validate_content(cls, value: object) -> str:
        return _require_nonblank_string(value, field_name="content")


class CitedAnswer(_StrictFrozenModel):
    # Keep the generated JSON Schema byte-contract free of a description field.
    answer: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("answer", mode="before")
    @classmethod
    def _validate_answer(cls, value: object) -> str:
        return _require_nonblank_string(value, field_name="answer")

    @field_validator("evidence_ids")
    @classmethod
    def _validate_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not evidence_id.strip() for evidence_id in value):
            raise ValueError("evidence_ids must contain only nonblank strings")
        if len(set(value)) != len(value):
            raise ValueError("evidence_ids must be unique while preserving order")
        return value


class StructuredGenerationRequest(_StrictFrozenModel):
    """Engine-neutral request for one schema-constrained generation."""

    messages: tuple[ModelMessage, ...] = Field(min_length=1)
    json_schema: _FrozenJsonObject = Field(min_length=1)
    schema_name: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    seed: int
    chat_template_kwargs: _FrozenJsonObject

    @field_validator("json_schema", mode="before")
    @classmethod
    def _isolate_json_schema(cls, value: object) -> dict[str, JsonValue]:
        return _clone_json_object(value, field_name="json_schema")

    @field_validator("schema_name", mode="before")
    @classmethod
    def _validate_schema_name(cls, value: object) -> str:
        return _require_nonblank_string(value, field_name="schema_name")

    @field_validator("temperature", mode="before")
    @classmethod
    def _validate_temperature(cls, value: object) -> float:
        return _require_finite_nonnegative_float(value, field_name="temperature")

    @field_validator("chat_template_kwargs", mode="before")
    @classmethod
    def _isolate_chat_template_kwargs(cls, value: object) -> dict[str, JsonValue]:
        return _clone_json_object(value, field_name="chat_template_kwargs")


class ModelTimings(_StrictFrozenModel):
    """Normalized finite timing measurements for one generation."""

    first_token_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)
    tokens_per_second: float = Field(ge=0.0)

    @field_validator("first_token_ms", "total_ms", "tokens_per_second", mode="before")
    @classmethod
    def _validate_finite_values(cls, value: object, info: object) -> float:
        field_name = getattr(info, "field_name", "timing")
        return _require_finite_nonnegative_float(value, field_name=field_name)

    @model_validator(mode="after")
    def _validate_timing_order(self) -> Self:
        if self.first_token_ms > self.total_ms:
            raise ValueError("first_token_ms cannot exceed total_ms")
        return self


class StructuredGenerationResult(_StrictFrozenModel):
    """Completed structured generation plus normalized usage and timings."""

    content: str = Field(min_length=1)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    timings: ModelTimings

    @field_validator("content", mode="before")
    @classmethod
    def _validate_content(cls, value: object) -> str:
        return _require_nonblank_string(value, field_name="content")

    @model_validator(mode="after")
    def _validate_token_counts(self) -> Self:
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        if self.prompt_tokens == 0:
            raise ValueError("completed generation requires positive prompt token usage")
        if self.completion_tokens == 0:
            raise ValueError("completed generation requires positive completion token usage")
        if self.timings.total_ms == 0.0:
            raise ValueError("completed generation requires positive total duration")
        if self.timings.first_token_ms == 0.0:
            raise ValueError("completed generation requires positive first-token latency")
        if self.timings.tokens_per_second == 0.0:
            raise ValueError("completed generation requires positive token rate")
        return self


@runtime_checkable
class CancellationSignal(Protocol):
    """Cooperative cancellation signal without runtime-specific state."""

    def is_set(self) -> bool: ...


@runtime_checkable
class StructuredLocalModel(Protocol):
    """Generate one schema-constrained result through a local model."""

    def generate(
        self,
        request: StructuredGenerationRequest,
        *,
        cancel: CancellationSignal,
    ) -> StructuredGenerationResult: ...
