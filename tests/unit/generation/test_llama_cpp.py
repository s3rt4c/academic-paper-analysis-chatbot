from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

import academic_chatbot.feasibility.llama_slice as llama_slice
import academic_chatbot.generation.llama_cpp as llama_cpp
from academic_chatbot.evidence.serialization import canonical_json_bytes
from academic_chatbot.evidence.service import EvidenceBundleService
from academic_chatbot.generation.context import render_generation_context
from academic_chatbot.generation.llama_cpp import (
    ContextLimitExceeded,
    GenerationCancelled,
    GenerationInvariantFailure,
    GenerationRuntimeUnavailable,
    GenerationTimeout,
    Phase0VerifiedSessionRunner,
    VerifiedGenerationResult,
    build_generation_body,
    count_and_admit,
    validate_frozen_manifests,
)
from academic_chatbot.ports.model import ModelTimings, StructuredGenerationResult
from tests.unit.evidence.test_packing import group, resolved


class _Resolver:
    def __init__(self, value: object) -> None:
        self.value = value

    def resolve(self, request: object) -> object:
        return self.value


def _ready_bundle():
    data = resolved(group())
    return EvidenceBundleService(resolver=_Resolver(data)).build(data.request)  # type: ignore[arg-type]


class _Cancel:
    def __init__(self, value: bool = False) -> None:
        self.value = value

    def is_set(self) -> bool:
        return self.value


class _Session:
    def __init__(
        self,
        *,
        tokens: int = 12,
        prompt: str = "rendered prompt",
        result_prompt_tokens: int | None = None,
    ) -> None:
        self.tokens = tokens
        self.prompt = prompt
        self.result_prompt_tokens = result_prompt_tokens
        self.applied: list[bytes] = []
        self.tokenized: list[bytes] = []
        self.generated: list[bytes] = []

    def apply_template(self, body: bytes) -> bytes:
        self.applied.append(body)
        return canonical_json_bytes({"prompt": self.prompt})

    def tokenize(self, body: bytes) -> bytes:
        self.tokenized.append(body)
        return canonical_json_bytes({"tokens": list(range(self.tokens))})

    def generate(self, body: bytes, *, cancel: _Cancel) -> StructuredGenerationResult:
        self.generated.append(body)
        if cancel.is_set():
            raise GenerationCancelled
        prompt_tokens = (
            self.tokens
            if self.result_prompt_tokens is None
            else self.result_prompt_tokens
        )
        return StructuredGenerationResult(
            content='{"answer":"A","citation_labels":["E1"]}',
            prompt_tokens=prompt_tokens,
            completion_tokens=4,
            total_tokens=prompt_tokens + 4,
            timings=ModelTimings(first_token_ms=1.0, total_ms=2.0, tokens_per_second=2.0),
        )


def _request():
    return render_generation_context(
        question="What is the objective?", bundle=_ready_bundle()
    ).request


_TEMPLATE_HASH = sha256(b"frozen template").hexdigest()


def test_generation_body_is_canonical_and_frozen() -> None:
    request = _request()
    body = build_generation_body(request)
    payload = json.loads(body)
    assert body == canonical_json_bytes(payload)
    assert payload["model"] == "local-academic"
    assert payload["cache_prompt"] is False
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_unvalidated_manifest_pair_is_rejected() -> None:
    with pytest.raises(GenerationRuntimeUnavailable):
        validate_frozen_manifests(
            runtime_manifest=object(),  # type: ignore[arg-type]
            model_manifest=object(),  # type: ignore[arg-type]
        )


def test_exact_context_boundary_is_closed() -> None:
    admitted = count_and_admit(_Session(tokens=3072), _request(), template_hash=_TEMPLATE_HASH)
    assert admitted.exact_prompt_token_count.token_count == 3072
    with pytest.raises(ContextLimitExceeded):
        count_and_admit(_Session(tokens=3073), _request(), template_hash=_TEMPLATE_HASH)


def test_template_and_tokenizer_are_bound_to_exact_request() -> None:
    session = _Session(tokens=2, prompt="p\N{LATIN SMALL LETTER E WITH ACUTE}")
    admitted = count_and_admit(session, _request(), template_hash=_TEMPLATE_HASH)
    exact = admitted.exact_prompt_token_count
    assert session.applied == [admitted.generation_body]
    assert json.loads(session.tokenized[0]) == {
        "add_special": True,
        "content": "p\N{LATIN SMALL LETTER E WITH ACUTE}",
        "parse_special": True,
        "with_pieces": False,
    }
    assert exact.complete_prompt_sha256 == sha256(
        "p\N{LATIN SMALL LETTER E WITH ACUTE}".encode()
    ).hexdigest()
    assert exact.generation_profile == "mvp-qwen3-8b-b10007-cuda-v1"
    assert exact.tokenizer_profile == "qwen3-8b-q4-k-m"


@pytest.mark.parametrize(
    "apply_body,token_body",
    [
        (b'{}', None),
        (b'{"prompt":"p","extra":1}', None),
        (b'{"prompt":"p","prompt":"q"}', None),
        (b'{"prompt":1}', None),
        (b'{"prompt":"p"}', b'{}'),
        (b'{"prompt":"p"}', b'{"tokens":[true]}'),
        (b'{"prompt":"p"}', b'{"tokens":[1],"extra":1}'),
        (b'{"prompt":"p"}', b'{"tokens":[1],"tokens":[2]}'),
    ],
)
def test_control_responses_are_strict(apply_body: bytes, token_body: bytes | None) -> None:
    class Bad(_Session):
        def apply_template(self, body: bytes) -> bytes:
            return apply_body

        def tokenize(self, body: bytes) -> bytes:
            return token_body or super().tokenize(body)

    with pytest.raises(GenerationRuntimeUnavailable):
        count_and_admit(Bad(), _request(), template_hash=_TEMPLATE_HASH)


def test_generate_sends_unchanged_body_and_retains_exact_count() -> None:
    session = _Session(tokens=7)
    admitted = count_and_admit(session, _request(), template_hash=_TEMPLATE_HASH)
    result = admitted.generate(cancel=_Cancel())
    assert isinstance(result, VerifiedGenerationResult)
    assert session.generated == [admitted.generation_body]
    assert result.exact_prompt_token_count.token_count == result.prompt_tokens == 7


def test_usage_mismatch_releases_no_result() -> None:
    admitted = count_and_admit(
        _Session(tokens=7, result_prompt_tokens=8),
        _request(),
        template_hash=_TEMPLATE_HASH,
    )
    with pytest.raises(GenerationInvariantFailure):
        admitted.generate(cancel=_Cancel())


@pytest.mark.parametrize("max_tokens", [1023, 1025])
def test_non_frozen_output_reservation_is_rejected(max_tokens: int) -> None:
    request = _request().model_copy(update={"max_tokens": max_tokens})
    with pytest.raises(GenerationRuntimeUnavailable):
        count_and_admit(
            _Session(), request, template_hash=_TEMPLATE_HASH
        )


def test_pre_cancel_starts_no_control_or_generation() -> None:
    session = _Session()
    with pytest.raises(GenerationCancelled):
        count_and_admit(
            session, _request(), template_hash=_TEMPLATE_HASH, cancel=_Cancel(True)
        )
    assert session.applied == session.tokenized == session.generated == []


def test_cancelled_generation_quarantines_partial_result() -> None:
    cancel = _Cancel()

    class During(_Session):
        def generate(
            self, body: bytes, *, cancel: _Cancel
        ) -> StructuredGenerationResult:
            result = super().generate(body, cancel=cancel)
            cancel.value = True
            return result

    session = During()
    admitted = count_and_admit(session, _request(), template_hash=_TEMPLATE_HASH)
    with pytest.raises(GenerationCancelled):
        admitted.generate(cancel=cancel)
    assert session.generated == [admitted.generation_body]


@pytest.mark.parametrize(
    "error",
    [
        ContextLimitExceeded(),
        GenerationCancelled(),
        GenerationInvariantFailure(),
        GenerationRuntimeUnavailable(),
        GenerationTimeout(),
    ],
)
def test_phase0_runner_preserves_frozen_adapter_error(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(**kwargs: object) -> object:
        del kwargs
        raise error

    monkeypatch.setattr(llama_cpp, "run_verified_llama_operation", fail)
    runner = Phase0VerifiedSessionRunner(
        runtime_directory=Path("runtime"),
        runtime_manifest=object(),  # type: ignore[arg-type]
        model_path=Path("model.gguf"),
        model_manifest=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(type(error)):
        runner.run(lambda _session: pytest.fail("operation must not run"))


def test_phase0_runner_maps_http_timeout_without_nested_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**kwargs: object) -> object:
        del kwargs
        raise llama_cpp.LlamaSliceHttpError("read_timeout")

    monkeypatch.setattr(llama_cpp, "run_verified_llama_operation", fail)
    runner = Phase0VerifiedSessionRunner(
        runtime_directory=Path("synthetic-runtime"),
        runtime_manifest=object(),  # type: ignore[arg-type]
        model_path=Path("synthetic-model.gguf"),
        model_manifest=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(GenerationTimeout) as raised:
        runner.run(lambda _session: pytest.fail("operation must not run"))
    assert str(raised.value) == ""


def test_real_sse_timeout_runs_recovery_and_maps_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Stream:
        status_code = 200

        def read(self, maximum_bytes: int, /) -> bytes:
            del maximum_bytes
            raise llama_slice.LlamaSseStreamTimeout("synthetic timeout detail")

        def close(self) -> None:
            events.append("close")

    class Transport:
        def open_chat_completion(self, body: bytes) -> Stream:
            assert body == b"request"
            return Stream()

    class Clock:
        def now_ns(self) -> int:
            return 1

    class Wait:
        def wait(self, seconds: float) -> None:
            del seconds

    def recover(**kwargs: object) -> None:
        del kwargs
        events.append("recover")

    monkeypatch.setattr(llama_cpp, "recover_llama_after_disconnect", recover)
    session = llama_cpp._Phase0PromptSession(
        transport=Transport(),  # type: ignore[arg-type]
        version=llama_slice.LlamaServerVersion(commit_prefix="00e79f6"),
        clock=Clock(),  # type: ignore[arg-type]
        wait_strategy=Wait(),  # type: ignore[arg-type]
    )

    with pytest.raises(GenerationTimeout) as raised:
        session.generate(b"request", cancel=_Cancel())
    assert str(raised.value) == ""
    assert events == ["recover", "close"]
