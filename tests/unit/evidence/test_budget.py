import json

import pytest

from academic_chatbot.evidence import models as m
from academic_chatbot.evidence.budget import measure_preview_bytes, render_content_preview


@pytest.mark.parametrize(
    "text", ["ASCII", "Türkçe", "漢字", "😀", '"quote"', "back\\slash", "line\n\t\x01"]
)
def test_exact_utf8_full_object_accounting(text):
    preview = m.ContentPreview(
        evidence=(m.ContentEvidence(citation_label="E1", physical_page_index=0, text=text),),
        context=(),
    )
    expected = json.dumps(
        preview.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    assert measure_preview_bytes(preview) == len(expected) > len(text.encode())


def test_empty_wrapper_is_77_bytes():
    assert measure_preview_bytes(render_content_preview((), ())) == 77
    with pytest.raises(ValueError):
        m.PreviewBudget(max_content_bytes=76)


def test_preview_projection_has_only_frozen_fields():
    from tests.unit.evidence.test_packing import group, pack

    result = pack(group(semantic=True, only=True))
    preview = render_content_preview(result.entries, result.context).model_dump(mode="json")
    assert set(preview) == {"concern_profile_id", "context", "evidence"}
    assert set(preview["evidence"][0]) == {"citation_label", "physical_page_index", "text", "trust"}
    assert set(preview["context"][0]) == {
        "related_citation_labels",
        "physical_page_index",
        "text",
        "trust",
        "citable",
    }
    assert preview["context"][0]["citable"] is False


def test_future_token_result_binds_complete_prompt_hash():
    from academic_chatbot.evidence.budget import ExactPromptTokenCount

    with pytest.raises(ValueError):
        ExactPromptTokenCount(
            complete_rendered_prompt="complete prompt",
            complete_prompt_sha256="a" * 64,
            generation_profile="generation",
            tokenizer_profile="tokenizer",
            template_hash="b" * 64,
            token_count=2,
        )


def test_fake_counter_is_only_an_interface_double():
    from hashlib import sha256

    from academic_chatbot.evidence.budget import ExactPromptTokenCount, ExactPromptTokenCounter

    class Fake:
        def count(
            self,
            complete_rendered_prompt,
            expected_generation_profile,
            expected_tokenizer_profile,
            expected_template_hash,
        ):
            return ExactPromptTokenCount(
                complete_rendered_prompt=complete_rendered_prompt,
                complete_prompt_sha256=sha256(complete_rendered_prompt.encode()).hexdigest(),
                generation_profile=expected_generation_profile,
                tokenizer_profile=expected_tokenizer_profile,
                template_hash=expected_template_hash,
                token_count=7,
            )

    counter: ExactPromptTokenCounter = Fake()
    result = counter.count("whole synthetic prompt", "generation", "tokenizer", "b" * 64)
    assert result.token_count == 7
    assert result.complete_rendered_prompt == "whole synthetic prompt"
    with pytest.raises(ValueError):
        result.token_count = 8
