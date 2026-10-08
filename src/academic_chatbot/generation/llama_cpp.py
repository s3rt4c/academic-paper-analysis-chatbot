"""Verified llama.cpp adapter for the frozen MVP generation profile."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol, cast

from pydantic import ValidationError

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.budget import ExactPromptTokenCount
from academic_chatbot.evidence.serialization import canonical_json_bytes
from academic_chatbot.feasibility.llama_slice import (
    GgufModelManifest,
    LlamaHttpxLoopbackTransport,
    LlamaMonotonicClock,
    LlamaRuntimeManifest,
    LlamaServerVersion,
    LlamaSliceCancellationError,
    LlamaSliceHttpError,
    LlamaSliceLifecycleError,
    LlamaSliceResponseError,
    LlamaSliceStartupError,
    LlamaWaitStrategy,
    parse_llama_chat_completion_stream_for_generation,
    recover_llama_after_disconnect,
    run_verified_llama_operation,
)
from academic_chatbot.generation.context import (
    MVP_CITED_ANSWER_JSON_SCHEMA,
    MVP_SYSTEM_MESSAGE,
)
from academic_chatbot.generation.models import (
    MVP_GENERATION_PROFILE_ID,
    MVP_MODEL_PROFILE_ID,
)
from academic_chatbot.ports.model import (
    CancellationSignal,
    StructuredGenerationRequest,
    StructuredGenerationResult,
)

MVP_RUNTIME_PROFILE_ID = "b10007-win-cuda-12.4-x64"
MVP_RUNTIME_MANIFEST_SHA256 = (
    "1a2bb81fbc106ada66bfdb66e1363b36fc82711ae8ccb19ca6791ae074aeb470"
)
MVP_RUNTIME_BUNDLE_SHA256 = (
    "43847d103a77ffc89e559a81483a7ccf7d489dccd215b7e892af5c5630cac912"
)
MVP_MODEL_MANIFEST_SHA256 = (
    "3ea914a437f122251c04102d8ef8766cfc57bff1f1d5211ed8c2a3359e22d4ae"
)
MVP_TOKENIZER_METADATA_SHA256 = (
    "288ed9dc7c26c09d6b861f6974cd1fcf67c863c92992bc14ab87eefc3d36ca66"
)
MVP_CONTEXT_TOKENS = 4096
MVP_RESERVED_OUTPUT_TOKENS = 1024


class ContextLimitExceeded(Exception):
    """The exact server tokenizer proved that the request cannot fit."""


class GenerationCancelled(Exception):
    """Generation was cancelled and no partial result may be released."""


class GenerationRuntimeUnavailable(Exception):
    """The frozen runtime or one of its strict control responses is unavailable."""


class GenerationTimeout(Exception):
    """The bounded local generation operation timed out."""


class GenerationProcessFailure(Exception):
    """The contained local model process failed."""


class GenerationInvariantFailure(Exception):
    """A frozen postcondition failed; no model output is safe to release."""


class PromptAuthoritySession(Protocol):
    """Same-process template, tokenizer and generation surface."""

    def apply_template(self, body: bytes) -> bytes: ...

    def tokenize(self, body: bytes) -> bytes: ...

    def generate(
        self, body: bytes, *, cancel: CancellationSignal
    ) -> StructuredGenerationResult: ...


class VerifiedSessionRunner(Protocol):
    def run(
        self,
        operation: Callable[[PromptAuthoritySession], StructuredGenerationResult],
    ) -> StructuredGenerationResult: ...


class _Phase0PromptSession:
    def __init__(
        self,
        *,
        transport: LlamaHttpxLoopbackTransport,
        version: LlamaServerVersion,
        clock: LlamaMonotonicClock,
        wait_strategy: LlamaWaitStrategy,
    ) -> None:
        self._transport = transport
        self._version = version
        self._clock = clock
        self._wait_strategy = wait_strategy

    def apply_template(self, body: bytes) -> bytes:
        return self._transport.post_apply_template(body).body

    def tokenize(self, body: bytes) -> bytes:
        return self._transport.post_tokenize(body).body

    def generate(
        self, body: bytes, *, cancel: CancellationSignal
    ) -> StructuredGenerationResult:
        if cancel.is_set():
            raise GenerationCancelled
        started = self._clock.now_ns()
        stream = self._transport.open_chat_completion(body)
        stop = threading.Event()

        def cancel_watcher() -> None:
            while not stop.wait(0.01):
                if cancel.is_set():
                    try:
                        stream.close()
                    except Exception:
                        pass
                    return

        watcher = threading.Thread(
            target=cancel_watcher,
            name="mvp-llama-cancellation-watcher",
            daemon=True,
        )
        watcher.start()
        try:
            result = parse_llama_chat_completion_stream_for_generation(
                stream=stream,
                clock=self._clock,
                request_started_ns=started,
                expected_version=self._version,
            )
        except LlamaSliceHttpError as error:
            if cancel.is_set():
                recover_llama_after_disconnect(
                    transport=self._transport,
                    clock=self._clock,
                    wait_strategy=self._wait_strategy,
                )
                raise GenerationCancelled from None
            if error.code == "read_timeout":
                recover_llama_after_disconnect(
                    transport=self._transport,
                    clock=self._clock,
                    wait_strategy=self._wait_strategy,
                )
                raise GenerationTimeout from None
            raise
        except LlamaSliceResponseError as error:
            if cancel.is_set():
                recover_llama_after_disconnect(
                    transport=self._transport,
                    clock=self._clock,
                    wait_strategy=self._wait_strategy,
                )
                raise GenerationCancelled from None
            if error.code == "timeout":
                recover_llama_after_disconnect(
                    transport=self._transport,
                    clock=self._clock,
                    wait_strategy=self._wait_strategy,
                )
                raise GenerationTimeout from None
            raise
        except Exception:
            if cancel.is_set():
                recover_llama_after_disconnect(
                    transport=self._transport,
                    clock=self._clock,
                    wait_strategy=self._wait_strategy,
                )
                raise GenerationCancelled from None
            raise
        finally:
            stop.set()
            try:
                stream.close()
            finally:
                watcher.join(1.0)
        if watcher.is_alive():
            raise GenerationInvariantFailure
        if cancel.is_set():
            recover_llama_after_disconnect(
                transport=self._transport,
                clock=self._clock,
                wait_strategy=self._wait_strategy,
            )
            raise GenerationCancelled
        return result


class Phase0VerifiedSessionRunner:
    """Bind the MVP adapter to the proven Phase 0 artifact/session lifecycle."""

    def __init__(
        self,
        *,
        runtime_directory: Path,
        runtime_manifest: LlamaRuntimeManifest,
        model_path: Path,
        model_manifest: GgufModelManifest,
    ) -> None:
        self._runtime_directory = runtime_directory
        self._runtime_manifest = runtime_manifest
        self._model_path = model_path
        self._model_manifest = model_manifest

    def run(
        self,
        operation: Callable[[PromptAuthoritySession], StructuredGenerationResult],
    ) -> StructuredGenerationResult:
        try:
            result = run_verified_llama_operation(
                runtime_directory=self._runtime_directory,
                runtime_manifest=self._runtime_manifest,
                model_path=self._model_path,
                model_manifest=self._model_manifest,
                operation=lambda transport, version, clock, wait_strategy: operation(
                    _Phase0PromptSession(
                        transport=transport,
                        version=version,
                        clock=clock,
                        wait_strategy=wait_strategy,
                    )
                ),
            )
        except MemoryError:
            raise
        except GenerationCancelled:
            raise
        except (
            ContextLimitExceeded,
            GenerationInvariantFailure,
            GenerationProcessFailure,
            GenerationRuntimeUnavailable,
            GenerationTimeout,
        ):
            raise
        except LlamaSliceHttpError as error:
            if error.code in {"connect_timeout", "read_timeout", "write_timeout", "pool_timeout"}:
                raise GenerationTimeout from None
            raise GenerationProcessFailure from None
        except (
            LlamaSliceCancellationError,
            LlamaSliceResponseError,
            LlamaSliceLifecycleError,
        ):
            raise GenerationProcessFailure from None
        except LlamaSliceStartupError:
            raise GenerationRuntimeUnavailable from None
        except Exception:
            raise GenerationProcessFailure from None
        if not isinstance(result, StructuredGenerationResult):
            raise GenerationInvariantFailure
        return result


class VerifiedGenerationResult(StructuredGenerationResult):
    exact_prompt_token_count: ExactPromptTokenCount
    generation_request_sha256: m.Sha256


def build_generation_body(request: StructuredGenerationRequest) -> bytes:
    """Build the one canonical OpenAI-compatible body used by all three calls."""

    payload = request.model_dump(mode="json", warnings="error")
    return canonical_json_bytes(
        {
            "cache_prompt": False,
            "chat_template_kwargs": payload["chat_template_kwargs"],
            "max_tokens": request.max_tokens,
            "messages": payload["messages"],
            "model": "local-academic",
            "response_format": {
                "json_schema": {
                    "name": request.schema_name,
                    "schema": payload["json_schema"],
                    "strict": True,
                },
                "type": "json_schema",
            },
            "seed": request.seed,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": request.temperature,
        }
    )


def _strict_object(raw: bytes) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"), object_pairs_hook=unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError):
        raise GenerationRuntimeUnavailable from None
    if type(value) is not dict:
        raise GenerationRuntimeUnavailable
    return cast(dict[str, object], value)


def _render_prompt(session: PromptAuthoritySession, body: bytes) -> str:
    value = _strict_object(session.apply_template(body))
    if set(value) != {"prompt"} or type(value["prompt"]) is not str:
        raise GenerationRuntimeUnavailable
    return value["prompt"]


def _token_count(session: PromptAuthoritySession, prompt: str) -> int:
    body = canonical_json_bytes(
        {
            "add_special": True,
            "content": prompt,
            "parse_special": True,
            "with_pieces": False,
        }
    )
    value = _strict_object(session.tokenize(body))
    if set(value) != {"tokens"} or type(value["tokens"]) is not list:
        raise GenerationRuntimeUnavailable
    tokens = cast(list[object], value["tokens"])
    if any(type(token) is not int for token in tokens):
        raise GenerationRuntimeUnavailable
    return len(tokens)


@dataclass(frozen=True, slots=True)
class AdmittedGeneration:
    session: PromptAuthoritySession
    generation_body: bytes
    exact_prompt_token_count: ExactPromptTokenCount
    generation_request_sha256: str

    def generate(self, *, cancel: CancellationSignal) -> VerifiedGenerationResult:
        if cancel.is_set():
            raise GenerationCancelled
        result = self.session.generate(self.generation_body, cancel=cancel)
        if cancel.is_set():
            raise GenerationCancelled
        if result.prompt_tokens != self.exact_prompt_token_count.token_count:
            raise GenerationInvariantFailure
        try:
            return VerifiedGenerationResult(
                **result.model_dump(mode="python", warnings="error"),
                exact_prompt_token_count=self.exact_prompt_token_count,
                generation_request_sha256=self.generation_request_sha256,
            )
        except (ValidationError, ValueError, TypeError):
            raise GenerationInvariantFailure from None


def count_and_admit(
    session: PromptAuthoritySession,
    request: StructuredGenerationRequest,
    *,
    template_hash: str,
    cancel: CancellationSignal | None = None,
) -> AdmittedGeneration:
    """Render and count on one session, then enforce the exact closed boundary."""

    if cancel is not None and cancel.is_set():
        raise GenerationCancelled
    payload = request.model_dump(mode="json", warnings="error")
    if (
        request.max_tokens != MVP_RESERVED_OUTPUT_TOKENS
        or request.temperature != 0.0
        or request.seed != 424242
        or request.schema_name != "mvp_cited_answer"
        or payload["json_schema"] != MVP_CITED_ANSWER_JSON_SCHEMA
        or payload["chat_template_kwargs"] != {"enable_thinking": False}
        or len(request.messages) != 2
        or request.messages[0].role != "system"
        or request.messages[0].content != MVP_SYSTEM_MESSAGE
        or request.messages[1].role != "user"
    ):
        raise GenerationRuntimeUnavailable
    if (
        type(template_hash) is not str
        or re.fullmatch(r"[0-9a-f]{64}", template_hash) is None
    ):
        raise GenerationInvariantFailure from None
    body = build_generation_body(request)
    prompt = _render_prompt(session, body)
    if cancel is not None and cancel.is_set():
        raise GenerationCancelled
    token_count = _token_count(session, prompt)
    if token_count + MVP_RESERVED_OUTPUT_TOKENS > MVP_CONTEXT_TOKENS:
        raise ContextLimitExceeded
    try:
        exact = ExactPromptTokenCount(
            complete_rendered_prompt=prompt,
            complete_prompt_sha256=sha256(prompt.encode("utf-8")).hexdigest(),
            generation_profile=MVP_GENERATION_PROFILE_ID,
            tokenizer_profile=MVP_MODEL_PROFILE_ID,
            template_hash=template_hash,
            token_count=token_count,
        )
    except (ValidationError, ValueError, TypeError):
        raise GenerationInvariantFailure from None
    return AdmittedGeneration(
        session=session,
        generation_body=body,
        exact_prompt_token_count=exact,
        generation_request_sha256=sha256(body).hexdigest(),
    )


def validate_frozen_manifests(
    *, runtime_manifest: LlamaRuntimeManifest, model_manifest: GgufModelManifest
) -> str:
    """Fail closed unless both already-validated manifests are the frozen MVP pair."""

    try:
        runtime_manifest = LlamaRuntimeManifest.model_validate(
            runtime_manifest.model_dump(mode="python", warnings="error"), strict=True
        )
        model_manifest = GgufModelManifest.model_validate(
            model_manifest.model_dump(mode="python", warnings="error"), strict=True
        )
    except (AttributeError, TypeError, ValidationError, ValueError):
        raise GenerationRuntimeUnavailable from None
    if (
        runtime_manifest.runtime_id != MVP_RUNTIME_PROFILE_ID
        or runtime_manifest.manifest_sha256 != MVP_RUNTIME_MANIFEST_SHA256
        or runtime_manifest.bundle_sha256 != MVP_RUNTIME_BUNDLE_SHA256
        or model_manifest.profile_id != MVP_MODEL_PROFILE_ID
        or model_manifest.manifest_sha256 != MVP_MODEL_MANIFEST_SHA256
        or model_manifest.tokenizer_metadata_sha256 != MVP_TOKENIZER_METADATA_SHA256
    ):
        raise GenerationRuntimeUnavailable
    return sha256(model_manifest.tokenizer_metadata.chat_template.encode("utf-8")).hexdigest()


class VerifiedLlamaCppModel:
    """StructuredLocalModel backed by one verified same-process operation."""

    def __init__(
        self,
        *,
        runner: VerifiedSessionRunner,
        runtime_manifest: LlamaRuntimeManifest,
        model_manifest: GgufModelManifest,
    ) -> None:
        self._runner = runner
        self._template_hash = validate_frozen_manifests(
            runtime_manifest=runtime_manifest, model_manifest=model_manifest
        )

    def generate(
        self,
        request: StructuredGenerationRequest,
        *,
        cancel: CancellationSignal,
    ) -> StructuredGenerationResult:
        if cancel.is_set():
            raise GenerationCancelled

        def operation(session: PromptAuthoritySession) -> StructuredGenerationResult:
            admitted = count_and_admit(
                session,
                request,
                template_hash=self._template_hash,
                cancel=cancel,
            )
            return admitted.generate(cancel=cancel)

        return self._runner.run(operation)
