import pytest
from pydantic import ValidationError

from academic_chatbot.config import RuntimeProfile
from academic_chatbot.domain.enums import FinalFieldStatus, JobState
from academic_chatbot.domain.models import EvidenceSpan


def test_canonical_job_state_values_are_stable() -> None:
    assert [state.value for state in JobState] == [
        "created",
        "queued",
        "preparing",
        "running",
        "checkpointing",
        "pause_requested",
        "paused",
        "cancel_requested",
        "cancelled",
        "waiting_user",
        "retry_wait",
        "queued_for_recovery",
        "completed",
        "failed",
    ]


def test_unsupported_is_not_a_final_field_status() -> None:
    with pytest.raises(ValueError):
        FinalFieldStatus("unsupported")


def test_evidence_offsets_are_half_open_and_ordered() -> None:
    with pytest.raises(ValidationError):
        EvidenceSpan(
            evidence_id="ev_1",
            paper_id="paper_1",
            file_version_id="fv_1",
            physical_page_index=0,
            printed_page_label=None,
            char_start=12,
            char_end=12,
            text="x",
            text_sha256="0" * 64,
            extraction_method="native_text",
            quality=0.9,
        )


def test_default_runtime_profile_matches_approved_budget() -> None:
    profile = RuntimeProfile()
    assert profile.context_tokens == 4096
    assert profile.max_evidence_tokens == 2000
    assert profile.max_output_tokens == 1024
    assert profile.max_heavy_jobs == 1
    assert profile.peak_process_tree_mib == 12 * 1024
