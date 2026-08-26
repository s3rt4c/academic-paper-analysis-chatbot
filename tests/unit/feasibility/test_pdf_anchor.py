import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Any

import pdfplumber
import pytest
from PIL import Image
from pydantic import ValidationError
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[3]
GENERATOR_PATH = ROOT / "tests" / "fixtures" / "pdfs" / "generate_native_anchor.py"
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "pdfs" / "native_anchor.pdf"
HARDWARE_FACTS_PATH = ROOT / "benchmarks" / "results" / "hardware-facts.json"
ANCHOR_SENTENCE = "The anchor sentence reports an accuracy of 91.2 percent."
FILE_VERSION_ID = "fv-phase0-native-anchor-v1"
FIXED_MEASURED_AT_UTC = "2026-07-14T12:34:56Z"
EXPECTED_HARDWARE_FACTS_SHA256 = (
    "552f2a908edea933b1c4bc4b2a8b381513bdc627025be96f61090318d998782c"
)
EXPECTED_FIXTURE_SIZE_BYTES = 2_470
EXPECTED_FIXTURE_SHA256 = "2d9c30592721d5e27f39c6a047f4e10f2577868d0ac1ef836a81dbdb8180175e"
EXPECTED_CANONICAL_PAGE_TEXT = (
    "Deterministic Native PDF Fixture Physical page 2 of 2. "
    "The anchor sentence reports an accuracy of 91.2 percent. "
    "The sentence above is the sole exact anchor in this document. A-7"
)
EXPECTED_CANONICAL_PAGE_TEXT_SHA256 = (
    "1e8db89970b5bc9db55546d506859fd5f565fce71df0edd926a68d5924ca3ebf"
)
EXPECTED_ANCHOR_TEXT_SHA256 = (
    "13ae5b7b01af4390ac74497e4d6d4a435cc12c2a09584848b9ad04e65897adcf"
)
EXPECTED_ANCHOR_BOXES = (
    {
        "char_start": 55,
        "char_end": 58,
        "x0": 72.0,
        "top": 139.277,
        "x1": 90.953,
        "bottom": 150.277,
    },
    {
        "char_start": 59,
        "char_end": 65,
        "x0": 94.011,
        "top": 139.277,
        "x1": 127.638,
        "bottom": 150.277,
    },
    {
        "char_start": 66,
        "char_end": 74,
        "x0": 130.696,
        "top": 139.277,
        "x1": 175.334,
        "bottom": 150.277,
    },
    {
        "char_start": 75,
        "char_end": 82,
        "x0": 178.392,
        "top": 139.277,
        "x1": 212.624,
        "bottom": 150.277,
    },
    {
        "char_start": 83,
        "char_end": 85,
        "x0": 215.682,
        "top": 139.277,
        "x1": 227.914,
        "bottom": 150.277,
    },
    {
        "char_start": 86,
        "char_end": 94,
        "x0": 230.972,
        "top": 139.277,
        "x1": 274.983,
        "bottom": 150.277,
    },
    {
        "char_start": 95,
        "char_end": 97,
        "x0": 278.041,
        "top": 139.277,
        "x1": 287.215,
        "bottom": 150.277,
    },
    {
        "char_start": 98,
        "char_end": 102,
        "x0": 290.273,
        "top": 139.277,
        "x1": 311.679,
        "bottom": 150.277,
    },
    {
        "char_start": 103,
        "char_end": 111,
        "x0": 314.737,
        "top": 139.277,
        "x1": 354.48,
        "bottom": 150.277,
    },
)
EXPECTED_PARSER_PROFILE_PAYLOAD = {
    "profile_id": "pdfplumber-native-anchor-v1",
    "pdfplumber_version": "0.11.10",
    "normalization_profile_id": "nfc-unicode-whitespace-ascii-space-v1",
    "unicode_normalization": "NFC",
    "whitespace_rule": "maximal-unicode-runs-to-ascii-space-and-trim",
    "preserve_case": True,
    "preserve_punctuation": True,
    "preserve_symbols": True,
    "preserve_compatibility_characters": True,
    "dehyphenate": False,
    "empty_normalized_token_policy": "discard-only-empty-normalized-word-tokens",
    "match_mode": "exact-overlapping-substring",
    "occurrence_policy": "find-every-overlapping-exact-normalized-substring",
    "substring_boundary_policy": "may-start-or-end-inside-source-word",
    "match_cardinality_policy": "zero-none-one-anchor-many-error",
    "anchor_box_policy": (
        "ordered-unmerged-source-word-boxes-intersecting-[char_start,char_end)"
    ),
    "inside_word_box_span_policy": "retain-full-page-local-source-word-span",
    "offset_unit": "page-local-unicode-code-point-half-open",
    "x_tolerance": 3.0,
    "y_tolerance": 3.0,
    "x_tolerance_ratio": None,
    "y_tolerance_ratio": None,
    "keep_blank_chars": False,
    "use_text_flow": False,
    "line_dir": "ttb",
    "char_dir": "ltr",
    "split_at_punctuation": False,
    "expand_ligatures": True,
    "return_chars": False,
    "line_vertical_key": "round6((top+bottom)/2)",
    "line_cluster_tolerance_points": 3.0,
    "line_representative": "first-word-frozen",
    "candidate_sort_keys": [
        "vertical_key",
        "x0",
        "top",
        "original_extraction_index",
    ],
    "line_sort_keys": [
        "representative_vertical_key",
        "minimum_x0",
        "minimum_original_extraction_index",
    ],
    "word_sort_keys": ["x0", "top", "original_extraction_index"],
    "word_joiner_ascii_codepoint": 32,
    "coordinate_system": "pdf-points-top-left-origin",
    "coordinate_decimal_places": 6,
    "canonicalize_negative_zero": True,
    "footer_band_points": 72.0,
    "footer_center_tolerance_points": 18.0,
    "footer_label_regex": "^[A-Z]+-[0-9]+$",
    "footer_candidate_policy": "exactly-one-standalone-line",
    "footer_band_formula": "rounded_line_top>=page_height_points-72.0",
    "footer_center_formula": (
        "abs((rounded_line_x0+rounded_line_x1)/2-page_width_points/2)<=18.0"
    ),
    "footer_cardinality_policy": "zero-or-many-none-one-label",
}
EXPECTED_RENDER_PROFILE_PAYLOAD = {
    "profile_id": "pdfium-rgba-v1",
    "scale": 2.0,
    "dpi": 144,
    "additional_rotation_degrees": 0,
    "crop": [0, 0, 0, 0],
    "may_draw_forms": False,
    "draw_annots": False,
    "fill_color": [255, 255, 255, 255],
    "force_bitmap_format": "FPDFBitmap_BGRA",
    "rev_byteorder": True,
    "optimize_mode": None,
    "no_smoothtext": False,
    "no_smoothimage": False,
    "no_smoothpath": False,
    "force_halftone": False,
    "limit_image_cache": False,
    "extra_flags": 0,
    "color_scheme": None,
    "fill_to_stroke": False,
    "source_rotation_recorded": True,
    "dimension_rule": "ceil-source-page-points-times-scale",
    "packed_pixel_mode": "RGBA",
    "packed_layout": "row-major-tight-rgba8",
    "raw_hash_domain_utf8": "pdfium-rgba-v1\u0000",
    "raw_hash_dimension_encoding": "uint64be-width-then-height",
    "png_mode": "RGBA",
    "png_optimize": False,
    "png_compress_level": 9,
    "png_metadata_policy": "none",
}
EXPECTED_PARSER_PROFILE_SHA256 = (
    "c7c474a7031ac8d0dbdedf31a9f70532e5e23f0fe59b0bdf901b8c8a436267dc"
)
EXPECTED_RENDER_PROFILE_SHA256 = (
    "a6dd8d04c3a69f2c87ef7aa38fc9adcc9a0674112c51cc46248481c87a7756d6"
)
EXPECTED_PAGE_1_PIXEL_SIZE = (1_224, 1_584)
EXPECTED_PAGE_1_RGBA_BYTE_COUNT = 7_755_264
EXPECTED_PAGE_1_RGBA_SHA256 = (
    "5981aa587ef5712368840795711ae5c01179ae4b38e1e6820b6787945f42bd4d"
)
EXPECTED_PAGE_1_RENDER_SHA256 = (
    "51146fb529d05930636779d894047e637afac3d1b0a87238965fd59075003c32"
)
EXPECTED_PAGE_1_PNG_BYTE_COUNT = 46_285
EXPECTED_PAGE_1_PNG_SHA256 = (
    "b42698373697c3029da02294b78b1d9fb2624f346f85cfe6464684f9a4baa855"
)
EXPECTED_FILE_VERSION_BINDING_SHA256 = (
    "2497249fc3f3b35388554b98ab4c9dbf75b810e32d612c3b8a91380900e87e60"
)
EXPECTED_EVIDENCE_ID = (
    "ev-sha256-208ff8ced2f81e9c1f94fb71bff43ce8ce57acac00b8c358c2e2ff9912a7d98a"
)
EXPECTED_REFERENCE_PROFILE_PAYLOAD = {
    "profile_id": "phase0-native-pdf-anchor-v1",
    "fixture_profile_id": "reportlab-native-anchor-v1",
    "fixture_size_bytes": EXPECTED_FIXTURE_SIZE_BYTES,
    "fixture_sha256": EXPECTED_FIXTURE_SHA256,
    "page_count": 2,
    "file_version_id": FILE_VERSION_ID,
    "file_version_binding_sha256": EXPECTED_FILE_VERSION_BINDING_SHA256,
    "needle": ANCHOR_SENTENCE,
    "physical_page_index": 1,
    "display_page_number": 2,
    "printed_page_label": "A-7",
    "parser_profile": EXPECTED_PARSER_PROFILE_PAYLOAD,
    "parser_profile_sha256": EXPECTED_PARSER_PROFILE_SHA256,
    "renderer_profile": EXPECTED_RENDER_PROFILE_PAYLOAD,
    "renderer_profile_sha256": EXPECTED_RENDER_PROFILE_SHA256,
    "python_version": "3.12.13",
    "pdfminer_version": "20260107",
    "pdfplumber_version": "0.11.10",
    "pypdfium2_version": "5.11.0",
    "pdfium_version": "151.0.7920.0",
    "pillow_version": "12.3.0",
    "reportlab_version": "5.0.0",
    "page_width_points": 612.0,
    "page_height_points": 792.0,
    "source_page_rotation_degrees": 0,
    "canonical_page_text_sha256": EXPECTED_CANONICAL_PAGE_TEXT_SHA256,
    "char_start": 55,
    "char_end": 111,
    "anchor_text_sha256": EXPECTED_ANCHOR_TEXT_SHA256,
    "boxes_sha256": "4fb9553245ae187bfc2fb4e4e31f754c3334234d94c5d8c3bd589b50ff04ede0",
    "evidence_id": EXPECTED_EVIDENCE_ID,
    "pixel_mode": "RGBA",
    "pixel_width": EXPECTED_PAGE_1_PIXEL_SIZE[0],
    "pixel_height": EXPECTED_PAGE_1_PIXEL_SIZE[1],
    "rgba_byte_count": EXPECTED_PAGE_1_RGBA_BYTE_COUNT,
    "rgba_sha256": EXPECTED_PAGE_1_RGBA_SHA256,
    "render_sha256": EXPECTED_PAGE_1_RENDER_SHA256,
    "png_byte_count": EXPECTED_PAGE_1_PNG_BYTE_COUNT,
    "png_sha256": EXPECTED_PAGE_1_PNG_SHA256,
}
EXPECTED_REFERENCE_PROFILE_SHA256 = (
    "caa8bd6d1382ad7447c35a84e58475c721e87a20f7c76931f916dd7912aa817c"
)


def _fixture_generator() -> ModuleType:
    if not GENERATOR_PATH.is_file():
        pytest.fail("The deterministic native-PDF fixture generator is not implemented.")
    spec = importlib.util.spec_from_file_location("native_anchor_fixture_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        pytest.fail("The deterministic native-PDF fixture generator cannot be loaded.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pdf_anchor() -> ModuleType:
    try:
        return importlib.import_module("academic_chatbot.feasibility.pdf_anchor")
    except ImportError:
        pytest.fail("The native-PDF anchor locator is not implemented.")


def _frozen_runtime_tool_versions(module: ModuleType) -> dict[str, str]:
    reference_profile = module.DEFAULT_REFERENCE_PROFILE
    return {
        field: getattr(reference_profile, field)
        for field in module._runtime_tool_versions()
    }


@pytest.fixture(autouse=True)
def _inject_frozen_reference_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    frozen_versions = _frozen_runtime_tool_versions(module)
    monkeypatch.setattr(
        module,
        "_runtime_tool_versions",
        lambda: dict(frozen_versions),
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_file_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _create_report(
    module: ModuleType,
    *,
    source: Path = FIXTURE_PATH,
    hardware_facts: Path = HARDWARE_FACTS_PATH,
    file_version_id: str = FILE_VERSION_ID,
    needle: str = ANCHOR_SENTENCE,
) -> Any:
    return module.create_pdf_anchor_report(
        source=source,
        file_version_id=file_version_id,
        needle=needle,
        hardware_facts_path=hardware_facts,
        measured_at_utc=FIXED_MEASURED_AT_UTC,
    )


def _rehash_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: value for key, value in payload.items() if key != "report_sha256"}
    payload["report_sha256"] = _canonical_sha256(unsigned)
    return payload


def _set_nested(payload: dict[str, Any], path: str, value: object) -> None:
    parts = path.split(".")
    target: dict[str, Any] = payload
    for part in parts[:-1]:
        nested = target[part]
        assert isinstance(nested, dict)
        target = nested
    target[parts[-1]] = value


def _with_original_python_report_hash(report: Any) -> Any:
    payload = report.model_dump(mode="python", warnings=False)
    unsigned = {key: value for key, value in payload.items() if key != "report_sha256"}
    return report.model_copy(update={"report_sha256": _canonical_sha256(unsigned)})


def _forge_normalization_preserving_report(
    module: ModuleType,
    report: Any,
    forgery: str,
) -> Any:
    if forgery == "anchor_page_float":
        anchor = report.anchor.model_copy(
            update={"page_width_points": int(report.anchor.page_width_points)}
        )
        forged = report.model_copy(update={"anchor": anchor})
    elif forgery == "box_coordinate_float":
        first = report.anchor.boxes[0].model_copy(
            update={"x0": int(report.anchor.boxes[0].x0)}
        )
        boxes = (first, *report.anchor.boxes[1:])
        boxes_sha256 = _canonical_sha256(
            [box.model_dump(mode="python", warnings=False) for box in boxes]
        )
        identity = {
            "file_version_id": report.anchor.file_version_id,
            "pdf_sha256": report.anchor.source_pdf_sha256,
            "parser_profile_sha256": report.anchor.parser_profile_sha256,
            "physical_page_index": report.anchor.physical_page_index,
            "char_start": report.anchor.char_start,
            "char_end": report.anchor.char_end,
            "anchor_text_sha256": report.anchor.anchor_text_sha256,
            "boxes_sha256": boxes_sha256,
        }
        anchor = report.anchor.model_copy(
            update={
                "boxes": boxes,
                "boxes_sha256": boxes_sha256,
                "evidence_id": "ev-sha256-" + _canonical_sha256(identity),
            }
        )
        forged = report.model_copy(
            update={"anchor": anchor, "boxes_sha256": boxes_sha256}
        )
    elif forgery in {"profile_float", "profile_tuple_list"}:
        parser = report.profile.parser_profile.model_copy(
            update=(
                {"x_tolerance": int(report.profile.parser_profile.x_tolerance)}
                if forgery == "profile_float"
                else {
                    "candidate_sort_keys": list(
                        report.profile.parser_profile.candidate_sort_keys
                    )
                }
            )
        )
        parser_sha256 = _canonical_sha256(
            parser.model_dump(mode="python", warnings=False)
        )
        profile = report.profile.model_copy(
            update={
                "parser_profile": parser,
                "parser_profile_sha256": parser_sha256,
            }
        )
        profile_sha256 = _canonical_sha256(
            profile.model_dump(mode="python", warnings=False)
        )
        forged = report.model_copy(
            update={
                "profile": profile,
                "profile_sha256": profile_sha256,
                "parser_profile_sha256": parser_sha256,
            }
        )
    elif forgery == "render_profile_float":
        renderer = report.profile.renderer_profile.model_copy(
            update={"scale": int(report.profile.renderer_profile.scale)}
        )
        renderer_sha256 = _canonical_sha256(
            renderer.model_dump(mode="python", warnings=False)
        )
        profile = report.profile.model_copy(
            update={
                "renderer_profile": renderer,
                "renderer_profile_sha256": renderer_sha256,
            }
        )
        profile_sha256 = _canonical_sha256(
            profile.model_dump(mode="python", warnings=False)
        )
        forged = report.model_copy(
            update={
                "profile": profile,
                "profile_sha256": profile_sha256,
                "renderer_profile_sha256": renderer_sha256,
            }
        )
    elif forgery == "render_evidence_float":
        render = report.render.model_copy(update={"scale": int(report.render.scale)})
        forged = report.model_copy(update={"render": render})
    else:
        profile_payload = report.profile.model_dump(mode="python")
        forged = module.PdfAnchorReport.model_construct(
            **{**report.__dict__, "profile": profile_payload}
        )
    return _with_original_python_report_hash(forged)


def _write_text_pdf(path: Path, pages: tuple[tuple[str, ...], ...]) -> None:
    pdf = Canvas(str(path), pagesize=letter, invariant=1, pageCompression=0)
    for lines in pages:
        for index, line in enumerate(lines):
            pdf.drawString(72, 720 - index * 24, line)
        pdf.showPage()
    pdf.save()


def _word(
    text: str,
    *,
    x0: float,
    top: float,
    x1: float,
    bottom: float,
) -> dict[str, object]:
    return {"text": text, "x0": x0, "top": top, "x1": x1, "bottom": bottom}


def test_native_anchor_fixture_is_deterministic_and_frozen(tmp_path: Path) -> None:
    generator = _fixture_generator()
    first = tmp_path / "native-anchor-a.pdf"
    second = tmp_path / "native-anchor-b.pdf"

    generator.generate_pdf(first)
    generator.generate_pdf(second)

    first_bytes = first.read_bytes()
    assert first_bytes == second.read_bytes()
    assert FIXTURE_PATH.read_bytes() == first_bytes
    assert len(first_bytes) == EXPECTED_FIXTURE_SIZE_BYTES
    assert hashlib.sha256(first_bytes).hexdigest() == EXPECTED_FIXTURE_SHA256


def test_native_anchor_fixture_has_expected_pages_text_and_labels() -> None:
    with pdfplumber.open(FIXTURE_PATH) as document:
        assert len(document.pages) == 2
        page_text = tuple(page.extract_text() for page in document.pages)
        page_sizes = tuple((page.width, page.height) for page in document.pages)

    assert page_sizes == ((612, 792), (612, 792))
    assert ANCHOR_SENTENCE not in page_text[0]
    assert page_text[1].count(ANCHOR_SENTENCE) == 1
    assert "A-6" in page_text[0]
    assert "A-7" in page_text[1]


def test_documents_port_has_exact_minimal_signature_and_immutable_records() -> None:
    module = _pdf_anchor()
    documents = importlib.import_module("academic_chatbot.ports.documents")
    signature = inspect.signature(documents.PdfAnchorLocator.locate)

    assert tuple(signature.parameters) == (
        "self",
        "source",
        "file_version_id",
        "needle",
    )
    assert signature.parameters["file_version_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["needle"].kind is inspect.Parameter.KEYWORD_ONLY

    box = documents.PdfAnchorBox(
        char_start=0,
        char_end=4,
        x0=0.0,
        top=1.0,
        x1=2.0,
        bottom=3.0,
    )
    with pytest.raises(ValidationError):
        box.x0 = 1.0
    with pytest.raises(ValidationError):
        documents.PdfAnchorBox(
            char_start=0,
            char_end=4,
            x0=0.0,
            top=1.0,
            x1=2.0,
            bottom=3.0,
            unexpected=True,
        )
    assert module.PdfPlumberAnchorLocator.locate is not None


def test_task5_public_module_interfaces_have_exact_frozen_signatures() -> None:
    module = _pdf_anchor()
    documents = importlib.import_module("academic_chatbot.ports.documents")
    expected = {
        "compute_pdf_parser_profile_sha256": "(profile: 'PdfParserProfile') -> 'str'",
        "compute_pdf_render_profile_sha256": "(profile: 'PdfRenderProfile') -> 'str'",
        "compute_pdf_anchor_reference_profile_sha256": (
            "(profile: 'PdfAnchorReferenceProfile') -> 'str'"
        ),
        "render_pdf_page": (
            "(source: 'Path', *, physical_page_index: 'int') -> 'RenderedPdfPage'"
        ),
        "create_pdf_anchor_report": (
            "(*, source: 'Path', file_version_id: 'str', needle: 'str', "
            "hardware_facts_path: 'Path', measured_at_utc: 'str | None' = None) "
            "-> 'PdfAnchorReport'"
        ),
        "load_pdf_anchor_report": "(path: 'Path') -> 'PdfAnchorReport'",
        "write_pdf_anchor_report": (
            "(path: 'Path', report: 'PdfAnchorReport') -> 'None'"
        ),
        "verify_pdf_anchor_replay": (
            "(*, source: 'Path', report: 'PdfAnchorReport') -> 'None'"
        ),
        "main": "(argv: 'Sequence[str] | None' = None) -> 'int'",
    }

    assert issubclass(module.PdfAnchorOperationalError, ValueError)
    for name, frozen_signature in expected.items():
        assert str(inspect.signature(getattr(module, name))) == frozen_signature
    locator_signature = (
        "(self, source: 'Path', *, file_version_id: 'str', needle: 'str') "
        "-> 'NativePdfAnchor | None'"
    )
    assert str(inspect.signature(module.PdfPlumberAnchorLocator.locate)) == (
        locator_signature
    )
    assert str(inspect.signature(documents.PdfAnchorLocator.locate)) == (
        locator_signature
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Cafe\u0301", "Caf\u00e9"),
        ("  alpha\u2003\n\tbeta  ", "alpha beta"),
        ("Case, punctuation!", "Case, punctuation!"),
        ("\uff21", "\uff21"),
        ("state-\nof-the-art", "state- of-the-art"),
    ],
)
def test_normalization_is_nfc_and_preserves_exact_match_semantics(
    source: str, expected: str
) -> None:
    module = _pdf_anchor()

    assert module._normalize_anchor_text(source) == expected


def test_reference_fixture_locator_roundtrips_exact_anchor() -> None:
    module = _pdf_anchor()

    anchor = module.PdfPlumberAnchorLocator().locate(
        FIXTURE_PATH,
        file_version_id=FILE_VERSION_ID,
        needle=ANCHOR_SENTENCE,
    )

    assert anchor is not None
    assert anchor.file_version_id == FILE_VERSION_ID
    assert anchor.source_pdf_sha256 == EXPECTED_FIXTURE_SHA256
    assert anchor.physical_page_index == 1
    assert anchor.display_page_number == 2
    assert anchor.printed_page_label == "A-7"
    assert (anchor.page_width_points, anchor.page_height_points) == (612.0, 792.0)
    assert anchor.source_page_rotation_degrees == 0
    assert anchor.canonical_page_text == EXPECTED_CANONICAL_PAGE_TEXT
    assert (anchor.char_start, anchor.char_end) == (55, 111)
    assert anchor.anchor_text == ANCHOR_SENTENCE
    assert anchor.canonical_page_text_sha256 == EXPECTED_CANONICAL_PAGE_TEXT_SHA256
    assert anchor.anchor_text_sha256 == EXPECTED_ANCHOR_TEXT_SHA256
    assert tuple(box.model_dump(mode="json") for box in anchor.boxes) == (
        EXPECTED_ANCHOR_BOXES
    )


def test_parser_and_render_input_profiles_are_exact_frozen_and_self_hashed() -> None:
    module = _pdf_anchor()
    parser_profile = module.PdfParserProfile()
    render_profile = module.PdfRenderProfile()

    assert parser_profile.model_dump(mode="json") == EXPECTED_PARSER_PROFILE_PAYLOAD
    assert render_profile.model_dump(mode="json") == EXPECTED_RENDER_PROFILE_PAYLOAD
    assert module.compute_pdf_parser_profile_sha256(parser_profile) == (
        EXPECTED_PARSER_PROFILE_SHA256
    )
    assert module.compute_pdf_render_profile_sha256(render_profile) == (
        EXPECTED_RENDER_PROFILE_SHA256
    )
    with pytest.raises(ValidationError):
        module.PdfParserProfile.model_validate(
            {**EXPECTED_PARSER_PROFILE_PAYLOAD, "x_tolerance": 4.0}
        )
    with pytest.raises(ValidationError):
        module.PdfRenderProfile.model_validate(
            {**EXPECTED_RENDER_PROFILE_PAYLOAD, "draw_annots": True}
        )
    with pytest.raises(ValidationError):
        module.PdfParserProfile.model_validate(
            {**EXPECTED_PARSER_PROFILE_PAYLOAD, "unexpected": True}
        )
    with pytest.raises(ValidationError):
        parser_profile.x_tolerance = 4.0


@pytest.mark.parametrize(
    ("model_name", "field", "coercing_value"),
    [
        ("PdfParserProfile", "x_tolerance", "3.0"),
        (
            "PdfParserProfile",
            "candidate_sort_keys",
            ["vertical_key", "x0", "top", "original_extraction_index"],
        ),
        ("PdfRenderProfile", "scale", "2.0"),
        ("PdfRenderProfile", "crop", [0, 0, 0, 0]),
    ],
)
def test_profiles_reject_python_mode_type_coercion(
    model_name: str, field: str, coercing_value: object
) -> None:
    module = _pdf_anchor()
    profile_type = getattr(module, model_name)
    payload = profile_type().model_dump(mode="python")
    payload[field] = coercing_value

    with pytest.raises(ValidationError):
        profile_type.model_validate(payload)


@pytest.mark.parametrize(
    ("model_name", "field", "integer_value"),
    [
        ("PdfParserProfile", "x_tolerance", 3),
        ("PdfParserProfile", "y_tolerance", 3),
        ("PdfParserProfile", "line_cluster_tolerance_points", 3),
        ("PdfParserProfile", "footer_band_points", 72),
        ("PdfParserProfile", "footer_center_tolerance_points", 18),
        ("PdfRenderProfile", "scale", 2),
        ("PdfRenderEvidence", "scale", 2),
        ("PdfAnchorReferenceProfile", "page_width_points", 612),
        ("PdfAnchorReferenceProfile", "page_height_points", 792),
    ],
)
@pytest.mark.parametrize("validation_mode", ["python", "json"])
def test_semantically_exact_float_fields_reject_integer_inputs(
    model_name: str,
    field: str,
    integer_value: int,
    validation_mode: str,
) -> None:
    module = _pdf_anchor()
    model_type = getattr(module, model_name)
    if model_name == "PdfRenderEvidence":
        model = module.render_pdf_page(
            FIXTURE_PATH,
            physical_page_index=1,
        ).evidence
    else:
        model = model_type()

    payload = model.model_dump(mode="python")
    payload[field] = integer_value
    with pytest.raises(ValidationError):
        if validation_mode == "python":
            model_type.model_validate(payload)
        else:
            model_type.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("field", ["page_width_points", "page_height_points"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_reference_page_dimensions_reject_non_finite_floats(
    field: str,
    value: float,
) -> None:
    module = _pdf_anchor()
    payload = module.PdfAnchorReferenceProfile().model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError):
        module.PdfAnchorReferenceProfile.model_validate(payload)


def test_strict_profiles_remain_loadable_from_json_mode() -> None:
    module = _pdf_anchor()

    parser = module.PdfParserProfile.model_validate_json(
        json.dumps(EXPECTED_PARSER_PROFILE_PAYLOAD)
    )
    renderer = module.PdfRenderProfile.model_validate_json(
        json.dumps(EXPECTED_RENDER_PROFILE_PAYLOAD)
    )

    assert parser == module.PdfParserProfile()
    assert renderer == module.PdfRenderProfile()


def test_complete_reference_profile_is_exact_strict_frozen_and_self_hashed() -> None:
    module = _pdf_anchor()
    profile = module.PdfAnchorReferenceProfile()

    assert profile.model_dump(mode="json") == EXPECTED_REFERENCE_PROFILE_PAYLOAD
    assert module.DEFAULT_REFERENCE_PROFILE == profile
    assert module.compute_pdf_anchor_reference_profile_sha256(profile) == (
        EXPECTED_REFERENCE_PROFILE_SHA256
    )
    assert module.DEFAULT_REFERENCE_PROFILE_SHA256 == EXPECTED_REFERENCE_PROFILE_SHA256
    assert module.PdfAnchorReferenceProfile.model_validate_json(
        json.dumps(EXPECTED_REFERENCE_PROFILE_PAYLOAD)
    ) == profile

    with pytest.raises(ValidationError):
        module.PdfAnchorReferenceProfile.model_validate(
            {**profile.model_dump(mode="python"), "page_width_points": "612.0"}
        )
    with pytest.raises(ValidationError):
        module.PdfAnchorReferenceProfile.model_validate(
            {**profile.model_dump(mode="python"), "unexpected": True}
        )
    with pytest.raises(ValidationError):
        profile.page_width_points = 1.0


def test_word_order_uses_non_chaining_line_clusters_and_stable_ties() -> None:
    module = _pdf_anchor()
    non_chaining = module._canonicalize_extracted_words(
        (
            _word("A", x0=0.0, top=9.0, x1=1.0, bottom=11.0),
            _word("B", x0=10.0, top=11.9, x1=11.0, bottom=13.9),
            _word("C", x0=20.0, top=14.8, x1=21.0, bottom=16.8),
        ),
        page_width_points=100.0,
        page_height_points=100.0,
    )
    tied = module._canonicalize_extracted_words(
        (
            _word("first", x0=5.0, top=5.0, x1=10.0, bottom=10.0),
            _word("second", x0=5.0, top=5.0, x1=10.0, bottom=10.0),
        ),
        page_width_points=100.0,
        page_height_points=100.0,
    )

    assert tuple(line.text for line in non_chaining.lines) == ("A B", "C")
    assert non_chaining.canonical_text == "A B C"
    assert tuple(line.text for line in tied.lines) == ("first second",)
    assert tied.canonical_text == "first second"


def test_empty_normalized_word_tokens_are_the_only_discarded_tokens() -> None:
    module = _pdf_anchor()

    page = module._canonicalize_extracted_words(
        (
            _word("\u2003\n", x0=0.0, top=0.0, x1=2.0, bottom=2.0),
            _word("Visible", x0=3.0, top=0.0, x1=10.0, bottom=2.0),
        ),
        page_width_points=100.0,
        page_height_points=100.0,
    )

    assert page.canonical_text == "Visible"
    assert len(page.words) == 1
    assert page.words[0].original_extraction_index == 1


def test_word_coordinates_are_six_decimal_and_negative_zero_canonical() -> None:
    module = _pdf_anchor()

    page = module._canonicalize_extracted_words(
        (
            _word(
                "rounded",
                x0=-0.0000004,
                top=1.2345674,
                x1=12.3456786,
                bottom=20.0000004,
            ),
        ),
        page_width_points=100.0,
        page_height_points=100.0,
    )

    word = page.words[0]
    assert (word.x0, word.top, word.x1, word.bottom) == (
        0.0,
        1.234567,
        12.345679,
        20.0,
    )
    assert str(word.x0) == "0.0"
    assert (word.char_start, word.char_end) == (0, 7)


@pytest.mark.parametrize(
    ("words", "expected_label"),
    [
        (
            (_word("A-7", x0=42.0, top=28.0, x1=58.0, bottom=38.0),),
            "A-7",
        ),
        (
            (_word("A-7", x0=60.0, top=28.0, x1=76.0, bottom=38.0),),
            "A-7",
        ),
        (
            (_word("A-7", x0=42.0, top=27.9, x1=58.0, bottom=37.9),),
            None,
        ),
        (
            (
                _word("A-7", x0=60.000002, top=28.0, x1=76.000002, bottom=38.0),
            ),
            None,
        ),
        (
            (
                _word("A-7", x0=42.0, top=28.0, x1=58.0, bottom=38.0),
                _word("B-8", x0=42.0, top=42.0, x1=58.0, bottom=52.0),
            ),
            None,
        ),
        (
            (
                _word("A-7", x0=35.0, top=28.0, x1=50.0, bottom=38.0),
                _word("note", x0=52.0, top=28.0, x1=65.0, bottom=38.0),
            ),
            None,
        ),
    ],
)
def test_printed_label_uses_only_unique_centered_final_band_lines(
    words: tuple[dict[str, object], ...], expected_label: str | None
) -> None:
    module = _pdf_anchor()

    page = module._canonicalize_extracted_words(
        words,
        page_width_points=100.0,
        page_height_points=100.0,
    )

    assert page.printed_page_label == expected_label


def test_footer_cardinality_maps_zero_and_many_candidates_to_none() -> None:
    module = _pdf_anchor()
    no_candidate = module._canonicalize_extracted_words(
        (),
        page_width_points=100.0,
        page_height_points=100.0,
    )
    many_candidates = module._canonicalize_extracted_words(
        (
            _word("A-7", x0=42.0, top=28.0, x1=58.0, bottom=38.0),
            _word("B-8", x0=42.0, top=42.0, x1=58.0, bottom=52.0),
        ),
        page_width_points=100.0,
        page_height_points=100.0,
    )

    assert no_candidate.printed_page_label is None
    assert many_candidates.printed_page_label is None


def test_locator_passes_the_exact_pdfplumber_word_extraction_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    observed: list[dict[str, object]] = []
    real_extract_words = pdfplumber.page.Page.extract_words

    def capture_extract_words(
        page: pdfplumber.page.Page, **kwargs: object
    ) -> list[dict[str, Any]]:
        observed.append(kwargs)
        return real_extract_words(page, **kwargs)

    monkeypatch.setattr(pdfplumber.page.Page, "extract_words", capture_extract_words)

    anchor = module.PdfPlumberAnchorLocator().locate(
        FIXTURE_PATH,
        file_version_id=FILE_VERSION_ID,
        needle=ANCHOR_SENTENCE,
    )

    assert anchor is not None
    assert observed == [
        {
            "x_tolerance": 3.0,
            "y_tolerance": 3.0,
            "x_tolerance_ratio": None,
            "y_tolerance_ratio": None,
            "keep_blank_chars": False,
            "use_text_flow": False,
            "line_dir": "ttb",
            "char_dir": "ltr",
            "split_at_punctuation": False,
            "expand_ligatures": True,
            "return_chars": False,
        },
        {
            "x_tolerance": 3.0,
            "y_tolerance": 3.0,
            "x_tolerance_ratio": None,
            "y_tolerance_ratio": None,
            "keep_blank_chars": False,
            "use_text_flow": False,
            "line_dir": "ttb",
            "char_dir": "ltr",
            "split_at_punctuation": False,
            "expand_ligatures": True,
            "return_chars": False,
        },
    ]


def test_locator_hashes_exact_boxes_file_binding_and_evidence_identity() -> None:
    module = _pdf_anchor()
    anchor = module.PdfPlumberAnchorLocator().locate(
        FIXTURE_PATH,
        file_version_id=FILE_VERSION_ID,
        needle=ANCHOR_SENTENCE,
    )
    assert anchor is not None

    expected_binding = _canonical_sha256(
        {"file_version_id": FILE_VERSION_ID, "pdf_sha256": EXPECTED_FIXTURE_SHA256}
    )
    expected_boxes_hash = _canonical_sha256(EXPECTED_ANCHOR_BOXES)
    identity_payload = {
        "file_version_id": FILE_VERSION_ID,
        "pdf_sha256": EXPECTED_FIXTURE_SHA256,
        "parser_profile_sha256": EXPECTED_PARSER_PROFILE_SHA256,
        "physical_page_index": 1,
        "char_start": 55,
        "char_end": 111,
        "anchor_text_sha256": EXPECTED_ANCHOR_TEXT_SHA256,
        "boxes_sha256": expected_boxes_hash,
    }

    assert anchor.file_version_binding_sha256 == expected_binding
    assert anchor.parser_profile_sha256 == EXPECTED_PARSER_PROFILE_SHA256
    assert anchor.boxes_sha256 == expected_boxes_hash
    assert anchor.evidence_id == "ev-sha256-" + _canonical_sha256(identity_payload)


def test_exact_inside_word_match_retains_the_full_source_word_span(
    tmp_path: Path,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "inside-word.pdf"
    _write_text_pdf(source, (("prefixTARGETsuffix",),))

    anchor = module.PdfPlumberAnchorLocator().locate(
        source,
        file_version_id="fv-inside-word",
        needle="TARGET",
    )

    assert anchor is not None
    assert (anchor.char_start, anchor.char_end) == (6, 12)
    assert len(anchor.boxes) == 1
    assert (anchor.boxes[0].char_start, anchor.boxes[0].char_end) == (0, 18)


def test_intersecting_source_word_boxes_remain_ordered_and_unmerged() -> None:
    module = _pdf_anchor()

    anchor = module.PdfPlumberAnchorLocator().locate(
        FIXTURE_PATH,
        file_version_id=FILE_VERSION_ID,
        needle="he anchor sent",
    )

    assert anchor is not None
    assert (anchor.char_start, anchor.char_end) == (56, 70)
    assert tuple((box.char_start, box.char_end) for box in anchor.boxes) == (
        (55, 58),
        (59, 65),
        (66, 74),
    )


def test_occurrence_search_includes_every_overlapping_exact_substring() -> None:
    module = _pdf_anchor()

    assert module._overlapping_occurrences("aaaa", "aa") == (0, 1, 2)


@pytest.mark.parametrize(
    "needle",
    [
        ANCHOR_SENTENCE.lower(),
        ANCHOR_SENTENCE.removesuffix(".") + "!",
        ANCHOR_SENTENCE.replace("91.2", "91.3"),
    ],
)
def test_locator_does_not_casefold_drop_punctuation_or_fuzzy_match(needle: str) -> None:
    module = _pdf_anchor()

    assert (
        module.PdfPlumberAnchorLocator().locate(
            FIXTURE_PATH,
            file_version_id=FILE_VERSION_ID,
            needle=needle,
        )
        is None
    )


def test_locator_normalizes_unicode_whitespace_in_the_needle() -> None:
    module = _pdf_anchor()
    needle = "  The anchor sentence\u2003reports an accuracy of 91.2\npercent.  "

    anchor = module.PdfPlumberAnchorLocator().locate(
        FIXTURE_PATH,
        file_version_id=FILE_VERSION_ID,
        needle=needle,
    )

    assert anchor is not None
    assert anchor.anchor_text == ANCHOR_SENTENCE


def test_locator_returns_none_for_no_match() -> None:
    module = _pdf_anchor()

    assert (
        module.PdfPlumberAnchorLocator().locate(
            FIXTURE_PATH,
            file_version_id=FILE_VERSION_ID,
            needle="This exact sentence is absent.",
        )
        is None
    )


@pytest.mark.parametrize("placement", ["same_page", "across_pages", "overlapping"])
def test_locator_rejects_every_duplicate_occurrence(
    tmp_path: Path, placement: str
) -> None:
    module = _pdf_anchor()
    source = tmp_path / f"duplicate-{placement}.pdf"
    if placement == "same_page":
        pages = ((ANCHOR_SENTENCE, ANCHOR_SENTENCE),)
        needle = ANCHOR_SENTENCE
    elif placement == "across_pages":
        pages = ((ANCHOR_SENTENCE,), (ANCHOR_SENTENCE,))
        needle = ANCHOR_SENTENCE
    else:
        pages = (("aaaa",),)
        needle = "aa"
    _write_text_pdf(source, pages)

    with pytest.raises(
        module.AmbiguousAnchorError,
        match=r"^The normalized anchor text matched more than once\.$",
    ):
        module.PdfPlumberAnchorLocator().locate(
            source,
            file_version_id="fv-duplicate",
            needle=needle,
        )


@pytest.mark.parametrize(
    ("file_version_id", "needle", "message"),
    [
        (FILE_VERSION_ID, "\u2003\n\t", "The normalized anchor text must not be empty."),
        (" \n\t ", ANCHOR_SENTENCE, "file_version_id must not be empty."),
    ],
)
def test_locator_rejects_empty_required_identity_inputs(
    file_version_id: str, needle: str, message: str
) -> None:
    module = _pdf_anchor()

    with pytest.raises(ValueError, match=f"^{message.replace('.', '[.]')}$"):
        module.PdfPlumberAnchorLocator().locate(
            FIXTURE_PATH,
            file_version_id=file_version_id,
            needle=needle,
        )


def test_locator_rejects_non_pdf_bytes(tmp_path: Path) -> None:
    module = _pdf_anchor()
    source = tmp_path / "not-a-pdf.pdf"
    source.write_bytes(b"not a PDF")

    with pytest.raises(ValueError, match=r"^The source file is not a PDF\.$"):
        module.PdfPlumberAnchorLocator().locate(
            source,
            file_version_id="fv-invalid",
            needle="anything",
        )


def test_locator_closes_the_source_for_immediate_windows_rename_and_delete(
    tmp_path: Path,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    renamed = tmp_path / "renamed.pdf"
    source.write_bytes(FIXTURE_PATH.read_bytes())

    anchor = module.PdfPlumberAnchorLocator().locate(
        source,
        file_version_id=FILE_VERSION_ID,
        needle=ANCHOR_SENTENCE,
    )
    os.replace(source, renamed)
    renamed.unlink()

    assert anchor is not None
    assert not source.exists()
    assert not renamed.exists()


def test_native_anchor_records_reject_rehashed_cross_field_tampering() -> None:
    module = _pdf_anchor()
    documents = importlib.import_module("academic_chatbot.ports.documents")
    anchor = module.PdfPlumberAnchorLocator().locate(
        FIXTURE_PATH,
        file_version_id=FILE_VERSION_ID,
        needle=ANCHOR_SENTENCE,
    )
    assert anchor is not None
    payload = anchor.model_dump(mode="json")

    with pytest.raises(ValidationError):
        documents.NativePdfAnchor.model_validate({**payload, "display_page_number": 3})
    with pytest.raises(ValidationError):
        documents.NativePdfAnchor.model_validate({**payload, "char_end": 110})
    with pytest.raises(ValidationError):
        documents.NativePdfAnchor.model_validate(
            {**payload, "canonical_page_text_sha256": "0" * 64}
        )
    with pytest.raises(ValidationError):
        documents.NativePdfAnchor.model_validate(
            {**payload, "evidence_id": "ev-sha256-" + "0" * 64}
        )
    with pytest.raises(ValidationError):
        documents.NativePdfAnchor.model_validate({**payload, "unexpected": True})
    with pytest.raises(ValidationError):
        anchor.char_start = 0


@pytest.mark.parametrize(
    "changes",
    [
        {"x0": float("nan")},
        {"x1": float("inf")},
        {"x0": 2.0, "x1": 2.0},
        {"top": 3.0, "bottom": 3.0},
        {"x0": -0.0},
        {"x1": 1.0000001},
    ],
)
def test_anchor_boxes_require_finite_ordered_coordinates(changes: dict[str, float]) -> None:
    documents = importlib.import_module("academic_chatbot.ports.documents")
    payload = {
        "char_start": 0,
        "char_end": 1,
        "x0": 0.0,
        "top": 0.0,
        "x1": 1.0,
        "bottom": 1.0,
        **changes,
    }

    with pytest.raises(ValidationError):
        documents.PdfAnchorBox.model_validate(payload)


def test_reference_page_render_is_deterministic_and_frozen() -> None:
    module = _pdf_anchor()

    first = module.render_pdf_page(FIXTURE_PATH, physical_page_index=1)
    second = module.render_pdf_page(FIXTURE_PATH, physical_page_index=1)

    assert first == second
    assert first.physical_page_index == 1
    assert (first.page_width_points, first.page_height_points) == (612.0, 792.0)
    assert (first.pixel_width, first.pixel_height) == EXPECTED_PAGE_1_PIXEL_SIZE
    assert len(first.packed_rgba_bytes) == EXPECTED_PAGE_1_RGBA_BYTE_COUNT
    assert len(first.png_bytes) == EXPECTED_PAGE_1_PNG_BYTE_COUNT
    assert first.evidence.rgba_sha256 == EXPECTED_PAGE_1_RGBA_SHA256
    assert first.evidence.render_sha256 == EXPECTED_PAGE_1_RENDER_SHA256
    assert first.evidence.png_sha256 == EXPECTED_PAGE_1_PNG_SHA256


def test_render_evidence_has_exact_fixed_profile_and_hash_preimages() -> None:
    module = _pdf_anchor()

    rendered = module.render_pdf_page(FIXTURE_PATH, physical_page_index=1)
    evidence = rendered.evidence
    expected_render_preimage = (
        b"pdfium-rgba-v1\0"
        + rendered.pixel_width.to_bytes(8, "big")
        + rendered.pixel_height.to_bytes(8, "big")
        + rendered.packed_rgba_bytes
    )

    assert evidence.model_dump(mode="json") == {
        "renderer_profile_id": "pdfium-rgba-v1",
        "physical_page_index": 1,
        "source_page_rotation_degrees": 0,
        "scale": 2.0,
        "dpi": 144,
        "additional_rotation_degrees": 0,
        "draw_annotations": False,
        "draw_forms": False,
        "background_rgba": [255, 255, 255, 255],
        "pixel_mode": "RGBA",
        "pixel_width": EXPECTED_PAGE_1_PIXEL_SIZE[0],
        "pixel_height": EXPECTED_PAGE_1_PIXEL_SIZE[1],
        "rgba_byte_count": EXPECTED_PAGE_1_RGBA_BYTE_COUNT,
        "rgba_sha256": EXPECTED_PAGE_1_RGBA_SHA256,
        "render_sha256": EXPECTED_PAGE_1_RENDER_SHA256,
        "png_byte_count": EXPECTED_PAGE_1_PNG_BYTE_COUNT,
        "png_sha256": EXPECTED_PAGE_1_PNG_SHA256,
    }
    assert evidence.rgba_sha256 == hashlib.sha256(
        rendered.packed_rgba_bytes
    ).hexdigest()
    assert evidence.render_sha256 == hashlib.sha256(expected_render_preimage).hexdigest()
    assert evidence.png_sha256 == hashlib.sha256(rendered.png_bytes).hexdigest()

    with Image.open(BytesIO(rendered.png_bytes)) as png:
        assert png.mode == "RGBA"
        assert png.size == EXPECTED_PAGE_1_PIXEL_SIZE
        assert png.info == {}
        assert png.tobytes() == rendered.packed_rgba_bytes


def test_render_profiles_pass_every_pdfium_argument_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    observed: list[dict[str, object]] = []
    real_render = module.pdfium.PdfPage.render

    def capture_render(page: object, **kwargs: object) -> Any:
        observed.append(kwargs)
        return real_render(page, **kwargs)

    monkeypatch.setattr(module.pdfium.PdfPage, "render", capture_render)

    module.render_pdf_page(FIXTURE_PATH, physical_page_index=1)

    assert observed == [
        {
            "scale": 2.0,
            "rotation": 0,
            "crop": (0, 0, 0, 0),
            "may_draw_forms": False,
            "fill_color": (255, 255, 255, 255),
            "draw_annots": False,
            "force_bitmap_format": module.pdfium.raw.FPDFBitmap_BGRA,
            "rev_byteorder": True,
            "optimize_mode": None,
            "no_smoothtext": False,
            "no_smoothimage": False,
            "no_smoothpath": False,
            "force_halftone": False,
            "limit_image_cache": False,
            "extra_flags": 0,
            "color_scheme": None,
            "fill_to_stroke": False,
        }
    ]


def test_rendered_pages_are_distinct_and_own_their_closed_resource_bytes() -> None:
    module = _pdf_anchor()

    page_zero = module.render_pdf_page(FIXTURE_PATH, physical_page_index=0)
    page_one = module.render_pdf_page(FIXTURE_PATH, physical_page_index=1)

    assert page_zero.evidence.render_sha256 != page_one.evidence.render_sha256
    assert page_zero.evidence.png_sha256 != page_one.evidence.png_sha256
    assert isinstance(page_one.packed_rgba_bytes, bytes)
    assert isinstance(page_one.png_bytes, bytes)
    with pytest.raises(FrozenInstanceError):
        page_one.pixel_width = 1
    with pytest.raises(ValidationError):
        page_one.evidence.pixel_width = 1


@pytest.mark.parametrize(
    ("physical_page_index", "message"),
    [
        (-1, "physical_page_index must be non-negative."),
        (2, "physical_page_index is outside the PDF page range."),
    ],
)
def test_render_rejects_invalid_page_indexes(
    physical_page_index: int, message: str
) -> None:
    module = _pdf_anchor()

    with pytest.raises(ValueError, match=f"^{message.replace('.', '[.]')}$"):
        module.render_pdf_page(
            FIXTURE_PATH,
            physical_page_index=physical_page_index,
        )


def test_renderer_excludes_stride_padding_and_copies_before_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    events: list[str] = []
    row_zero = bytes(range(1, 9))
    row_one = bytes(range(9, 17))
    padding = b"\xfe\xfd\xfc\xfb"

    class FakeBitmap:
        width = 2
        height = 2
        stride = 12
        n_channels = 4

        def __init__(self) -> None:
            self.buffer = bytearray(row_zero + padding + row_one + padding)

        def close(self) -> None:
            events.append("bitmap_close")
            self.buffer[:] = b"\x00" * len(self.buffer)

    bitmap = FakeBitmap()

    class FakePage:
        def get_width(self) -> float:
            return 1.0

        def get_height(self) -> float:
            return 1.0

        def get_rotation(self) -> int:
            return 90

        def render(self, **kwargs: object) -> FakeBitmap:
            events.append("render")
            return bitmap

        def close(self) -> None:
            events.append("page_close")

    page = FakePage()

    class FakeDocument:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> FakePage:
            assert index == 0
            return page

        def close(self) -> None:
            events.append("document_close")

    document = FakeDocument()
    monkeypatch.setattr(module.pdfium, "PdfDocument", lambda data: document)
    snapshot_bytes = b"%PDF-fake"
    snapshot = module._PdfSnapshot(
        data=snapshot_bytes,
        size_bytes=len(snapshot_bytes),
        sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
    )

    rendered = module._render_pdf_page_snapshot(snapshot, physical_page_index=0)

    assert rendered.packed_rgba_bytes == row_zero + row_one
    assert padding not in rendered.packed_rgba_bytes
    assert rendered.evidence.source_page_rotation_degrees == 90
    assert rendered.evidence.rgba_byte_count == 16
    assert rendered.packed_rgba_bytes != bytes(bitmap.buffer[:16])
    assert events == ["render", "bitmap_close", "page_close", "document_close"]


@pytest.mark.parametrize("close_failure", ["bitmap", "page"])
def test_renderer_attempts_all_closures_in_order_when_an_earlier_close_fails(
    monkeypatch: pytest.MonkeyPatch,
    close_failure: str,
) -> None:
    module = _pdf_anchor()
    events: list[str] = []

    class FailingBitmap:
        width = 1
        height = 1
        stride = 4
        n_channels = 4
        buffer = bytearray(b"\x01\x02\x03\x04")

        def close(self) -> None:
            events.append("bitmap_close")
            if close_failure == "bitmap":
                raise RuntimeError("bitmap close failure")

    bitmap = FailingBitmap()

    class FailingPage:
        def get_width(self) -> float:
            return 0.5

        def get_height(self) -> float:
            return 0.5

        def get_rotation(self) -> int:
            return 0

        def render(self, **kwargs: object) -> FailingBitmap:
            events.append("render")
            return bitmap

        def close(self) -> None:
            events.append("page_close")
            if close_failure == "page":
                raise RuntimeError("page close failure")

    page = FailingPage()

    class FailingDocument:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> FailingPage:
            assert index == 0
            return page

        def close(self) -> None:
            events.append("document_close")

    monkeypatch.setattr(module.pdfium, "PdfDocument", lambda data: FailingDocument())
    snapshot_bytes = b"%PDF-close-failure"
    snapshot = module._PdfSnapshot(
        data=snapshot_bytes,
        size_bytes=len(snapshot_bytes),
        sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
    )

    with pytest.raises(RuntimeError, match=f"{close_failure} close failure"):
        module._render_pdf_page_snapshot(snapshot, physical_page_index=0)

    assert events == ["render", "bitmap_close", "page_close", "document_close"]


def test_render_closes_source_for_immediate_windows_rename_and_delete(
    tmp_path: Path,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    renamed = tmp_path / "renamed.pdf"
    source.write_bytes(FIXTURE_PATH.read_bytes())

    rendered = module.render_pdf_page(source, physical_page_index=1)
    os.replace(source, renamed)
    renamed.unlink()

    assert rendered.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert not source.exists()
    assert not renamed.exists()


def test_render_evidence_rejects_extra_coercing_and_inconsistent_fields() -> None:
    module = _pdf_anchor()
    rendered = module.render_pdf_page(FIXTURE_PATH, physical_page_index=1)
    payload = rendered.evidence.model_dump(mode="python")

    with pytest.raises(ValidationError):
        module.PdfRenderEvidence.model_validate({**payload, "unexpected": True})
    with pytest.raises(ValidationError):
        module.PdfRenderEvidence.model_validate({**payload, "scale": "2.0"})
    with pytest.raises(ValidationError):
        module.PdfRenderEvidence.model_validate(
            {**payload, "rgba_byte_count": payload["rgba_byte_count"] - 1}
        )


def test_report_factory_and_validation_require_frozen_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()

    report = _create_report(module)
    unsigned = report.model_dump(mode="json", exclude={"report_sha256"})

    assert report.schema_version == "1.0.0"
    assert report.report_type == "pdf_anchor"
    assert report.artifact_kind == "deterministic_correctness"
    assert report.verification_status == "verified"
    assert report.measured_at_utc == FIXED_MEASURED_AT_UTC
    assert report.profile == module.DEFAULT_REFERENCE_PROFILE
    assert report.profile_sha256 == EXPECTED_REFERENCE_PROFILE_SHA256
    assert report.parser_profile_sha256 == EXPECTED_PARSER_PROFILE_SHA256
    assert report.renderer_profile_sha256 == EXPECTED_RENDER_PROFILE_SHA256
    assert report.reference_profile_verified is True
    assert report.pdf_sha256 == EXPECTED_FIXTURE_SHA256
    assert report.pdf_size_bytes == EXPECTED_FIXTURE_SIZE_BYTES
    assert report.page_count == 2
    assert report.file_version_id == FILE_VERSION_ID
    assert report.file_version_binding_sha256 == EXPECTED_FILE_VERSION_BINDING_SHA256
    assert report.hardware_facts_sha256 == EXPECTED_HARDWARE_FACTS_SHA256
    assert (
        report.python_version,
        report.pdfminer_version,
        report.pdfplumber_version,
        report.pypdfium2_version,
        report.pdfium_version,
        report.pillow_version,
        report.reportlab_version,
    ) == (
        "3.12.13",
        "20260107",
        "0.11.10",
        "5.11.0",
        "151.0.7920.0",
        "12.3.0",
        "5.0.0",
    )
    assert report.anchor.evidence_id == EXPECTED_EVIDENCE_ID
    assert report.anchor.anchor_text == ANCHOR_SENTENCE
    assert report.boxes_sha256 == report.anchor.boxes_sha256
    assert report.render.render_sha256 == EXPECTED_PAGE_1_RENDER_SHA256
    assert report.render.png_sha256 == EXPECTED_PAGE_1_PNG_SHA256
    assert report.anchor_integrity_verified is True
    assert report.render_integrity_verified is True
    assert report.report_sha256 == _canonical_sha256(unsigned)

    with pytest.raises(ValidationError):
        report.pdf_size_bytes = 1
    with pytest.raises(ValidationError):
        module.PdfAnchorReport.model_validate(
            {**report.model_dump(mode="python"), "unexpected": True}
        )

    mismatched_versions = _frozen_runtime_tool_versions(module)
    mismatched_versions["python_version"] = "0.0.0"
    monkeypatch.setattr(
        module,
        "_runtime_tool_versions",
        lambda: dict(mismatched_versions),
    )

    with pytest.raises(
        module.PdfAnchorOperationalError,
        match="Installed PDF tool versions do not match the reference profile",
    ):
        _create_report(module)

    with pytest.raises(ValidationError, match="installed runtime"):
        module.PdfAnchorReport.model_validate(report.model_dump(mode="python"))


@pytest.mark.parametrize(
    ("field_path", "value"),
    [
        ("profile_sha256", "0" * 64),
        ("parser_profile_sha256", "0" * 64),
        ("renderer_profile_sha256", "0" * 64),
        ("reference_profile_verified", False),
        ("pdf_sha256", "0" * 64),
        ("pdf_size_bytes", EXPECTED_FIXTURE_SIZE_BYTES + 1),
        ("page_count", 3),
        ("file_version_id", "fv-other"),
        ("file_version_binding_sha256", "0" * 64),
        ("python_version", "3.12.12"),
        ("pdfminer_version", "old"),
        ("pdfplumber_version", "old"),
        ("pypdfium2_version", "old"),
        ("pdfium_version", "old"),
        ("pillow_version", "old"),
        ("reportlab_version", "old"),
        ("profile.fixture_sha256", "0" * 64),
        ("profile.parser_profile_sha256", "0" * 64),
        ("profile.parser_profile.x_tolerance", 4.0),
        ("profile.renderer_profile_sha256", "0" * 64),
        ("profile.renderer_profile.scale", 3.0),
        ("boxes_sha256", "0" * 64),
        ("anchor.printed_page_label", "A-8"),
        ("anchor.page_width_points", 611.0),
        ("anchor.page_height_points", 791.0),
        ("anchor.source_page_rotation_degrees", 90),
        ("anchor.char_start", 54),
        ("anchor.canonical_page_text", "tampered"),
        ("anchor.canonical_page_text_sha256", "0" * 64),
        ("anchor.anchor_text", "tampered"),
        ("anchor.anchor_text_sha256", "0" * 64),
        ("anchor.boxes", []),
        ("anchor.boxes_sha256", "0" * 64),
        ("anchor.evidence_id", "ev-sha256-" + "0" * 64),
        ("render.source_page_rotation_degrees", 90),
        ("render.pixel_mode", "RGB"),
        ("render.rgba_byte_count", EXPECTED_PAGE_1_RGBA_BYTE_COUNT - 4),
        ("render.rgba_sha256", "0" * 64),
        ("render.render_sha256", "0" * 64),
        ("render.png_byte_count", EXPECTED_PAGE_1_PNG_BYTE_COUNT - 1),
        ("render.png_sha256", "0" * 64),
        ("render.pixel_width", EXPECTED_PAGE_1_PIXEL_SIZE[0] + 1),
    ],
)
def test_rehashed_report_rejects_every_default_reference_cross_field_tamper(
    field_path: str,
    value: object,
) -> None:
    module = _pdf_anchor()
    payload = _create_report(module).model_dump(mode="json")
    _set_nested(payload, field_path, value)
    _rehash_report_payload(payload)

    with pytest.raises(ValidationError):
        module.PdfAnchorReport.model_validate_json(json.dumps(payload))


def test_frozen_report_rejects_fully_rehashed_alternative_canonical_page_text() -> None:
    module = _pdf_anchor()
    payload = _create_report(module).model_dump(mode="json")
    anchor = payload["anchor"]
    assert isinstance(anchor, dict)
    canonical_text = anchor["canonical_page_text"]
    assert isinstance(canonical_text, str)
    anchor["canonical_page_text"] = "X" + canonical_text[1:]
    anchor["canonical_page_text_sha256"] = hashlib.sha256(
        anchor["canonical_page_text"].encode("utf-8")
    ).hexdigest()
    _rehash_report_payload(payload)

    internally_valid = module.NativePdfAnchor.model_validate(anchor)

    assert internally_valid.anchor_text == ANCHOR_SENTENCE
    with pytest.raises(ValidationError, match="reference profile"):
        module.PdfAnchorReport.model_validate_json(json.dumps(payload))


def test_frozen_report_rejects_fully_rehashed_alternative_boxes_and_evidence() -> None:
    module = _pdf_anchor()
    payload = _create_report(module).model_dump(mode="json")
    anchor = payload["anchor"]
    assert isinstance(anchor, dict)
    boxes = anchor["boxes"]
    assert isinstance(boxes, list)
    first_box = boxes[0]
    assert isinstance(first_box, dict)
    first_box["x0"] = first_box["x0"] + 1.0
    boxes_sha256 = _canonical_sha256(boxes)
    anchor["boxes_sha256"] = boxes_sha256
    payload["boxes_sha256"] = boxes_sha256
    identity = {
        "file_version_id": anchor["file_version_id"],
        "pdf_sha256": anchor["source_pdf_sha256"],
        "parser_profile_sha256": anchor["parser_profile_sha256"],
        "physical_page_index": anchor["physical_page_index"],
        "char_start": anchor["char_start"],
        "char_end": anchor["char_end"],
        "anchor_text_sha256": anchor["anchor_text_sha256"],
        "boxes_sha256": boxes_sha256,
    }
    anchor["evidence_id"] = "ev-sha256-" + _canonical_sha256(identity)
    _rehash_report_payload(payload)

    internally_valid = module.NativePdfAnchor.model_validate(anchor)

    assert internally_valid.evidence_id == anchor["evidence_id"]
    with pytest.raises(ValidationError, match="reference profile"):
        module.PdfAnchorReport.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "measured_at_utc",
    ["", "2026-07-14", "2026-07-14T12:34:56+03:00", "not-a-timestamp"],
)
def test_report_rejects_non_utc_or_invalid_measurement_time(
    measured_at_utc: str,
) -> None:
    module = _pdf_anchor()

    with pytest.raises((ValidationError, ValueError)):
        module.create_pdf_anchor_report(
            source=FIXTURE_PATH,
            file_version_id=FILE_VERSION_ID,
            needle=ANCHOR_SENTENCE,
            hardware_facts_path=HARDWARE_FACTS_PATH,
            measured_at_utc=measured_at_utc,
        )


@pytest.mark.parametrize(
    "needle",
    [
        ANCHOR_SENTENCE.replace("The anchor", "The\u00a0anchor"),
        ANCHOR_SENTENCE.replace("sentence reports", "sentence\treports"),
        ANCHOR_SENTENCE.replace("accuracy of", "accuracy\u2003of"),
    ],
)
def test_factory_locates_but_never_publishes_normalization_equivalent_raw_needle(
    monkeypatch: pytest.MonkeyPatch,
    needle: str,
) -> None:
    module = _pdf_anchor()
    assert needle != ANCHOR_SENTENCE
    assert module._normalize_anchor_text(needle) == ANCHOR_SENTENCE

    def unexpected_render(*args: object, **kwargs: object) -> Any:
        raise AssertionError("A non-default raw needle must fail before rendering.")

    monkeypatch.setattr(module, "_render_pdf_page_snapshot", unexpected_render)

    with pytest.raises(
        module.PdfAnchorOperationalError,
        match="requested anchor text does not exactly match",
    ):
        _create_report(module, needle=needle)


def test_canonical_hardware_loader_returns_exact_payload_hash_without_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    hardware = importlib.import_module("academic_chatbot.feasibility.hardware")

    def fail_collection() -> object:
        raise AssertionError("Live hardware collection must not run in Task 5.")

    monkeypatch.setattr(hardware, "collect_windows_hardware", fail_collection)
    facts, payload_sha256 = module._load_canonical_hardware_facts(HARDWARE_FACTS_PATH)

    assert facts.schema_version == "1.0.0"
    assert payload_sha256 == EXPECTED_HARDWARE_FACTS_SHA256
    assert "collect_windows_hardware" not in module.__dict__
    assert "ReferenceHardwareRecord" not in module.__dict__
    assert _create_report(module).hardware_facts_sha256 == payload_sha256


def test_report_factory_derives_unpinned_hardware_hash_from_supplied_canonical_file(
    tmp_path: Path,
) -> None:
    module = _pdf_anchor()
    payload = json.loads(HARDWARE_FACTS_PATH.read_text(encoding="utf-8"))
    payload["collected_at"] = "2026-07-14T00:00:00Z"
    hardware = tmp_path / "hardware.json"
    raw = _canonical_json_file_bytes(payload)
    hardware.write_bytes(raw)

    report = _create_report(module, hardware_facts=hardware)

    assert report.hardware_facts_sha256 == hashlib.sha256(raw[:-1]).hexdigest()
    assert report.hardware_facts_sha256 != EXPECTED_HARDWARE_FACTS_SHA256


@pytest.mark.parametrize(
    "case",
    [
        "invalid_utf8",
        "not_object",
        "duplicate_key",
        "nested_duplicate_key",
        "nan",
        "infinity",
        "bom",
        "noncanonical_spacing",
        "missing_newline",
        "crlf",
        "extra_newline",
        "coercing_value",
        "coercing_boolean",
        "reference_record",
    ],
)
def test_hardware_loader_rejects_noncanonical_coercing_or_wrong_schema_files(
    tmp_path: Path,
    case: str,
) -> None:
    module = _pdf_anchor()
    target = tmp_path / "hardware.json"
    payload = json.loads(HARDWARE_FACTS_PATH.read_text(encoding="utf-8"))
    if case == "invalid_utf8":
        target.write_bytes(b"\xff\n")
    elif case == "not_object":
        target.write_bytes(b"[]\n")
    elif case == "duplicate_key":
        target.write_bytes(b'{"schema_version":"1.0.0","schema_version":"1.0.0"}\n')
    elif case == "nested_duplicate_key":
        raw = _canonical_json_file_bytes(payload)
        target.write_bytes(
            raw.replace(
                b'{"bank_label":',
                b'{"bank_label":"duplicate","bank_label":',
                1,
            )
        )
    elif case == "nan":
        target.write_bytes(b'{"physical_cores":NaN,"schema_version":"1.0.0"}\n')
    elif case == "infinity":
        target.write_bytes(
            b'{"physical_cores":Infinity,"schema_version":"1.0.0"}\n'
        )
    elif case == "bom":
        target.write_bytes(b"\xef\xbb\xbf" + _canonical_json_file_bytes(payload))
    elif case == "noncanonical_spacing":
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    elif case == "missing_newline":
        target.write_bytes(_canonical_json_file_bytes(payload).removesuffix(b"\n"))
    elif case == "crlf":
        target.write_bytes(_canonical_json_file_bytes(payload).removesuffix(b"\n") + b"\r\n")
    elif case == "extra_newline":
        target.write_bytes(_canonical_json_file_bytes(payload) + b"\n")
    elif case == "coercing_value":
        payload["physical_cores"] = "8"
        target.write_bytes(_canonical_json_file_bytes(payload))
    elif case == "coercing_boolean":
        payload["gpu_offload_available"] = "false"
        target.write_bytes(_canonical_json_file_bytes(payload))
    else:
        payload["record_sha256"] = "0" * 64
        target.write_bytes(_canonical_json_file_bytes(payload))

    with pytest.raises(ValueError):
        module._load_canonical_hardware_facts(target)


def test_report_writer_and_loader_use_exact_canonical_self_hashed_json(
    tmp_path: Path,
) -> None:
    module = _pdf_anchor()
    report = _create_report(module)
    output = tmp_path / "report.json"

    module.write_pdf_anchor_report(output, report)
    loaded = module.load_pdf_anchor_report(output)

    assert loaded == report
    assert output.read_bytes() == _canonical_json_file_bytes(
        report.model_dump(mode="json")
    )


@pytest.mark.parametrize(
    "case",
    [
        "wrong_hash_first",
        "duplicate_key",
        "nested_duplicate_key",
        "nan",
        "infinity",
        "bom",
        "noncanonical_spacing",
        "missing_newline",
        "crlf",
        "extra_newline",
        "coercing_value",
        "coercing_boolean",
    ],
)
def test_report_loader_rejects_raw_hash_duplicate_noncanonical_and_coercion(
    tmp_path: Path,
    case: str,
) -> None:
    module = _pdf_anchor()
    payload = _create_report(module).model_dump(mode="json")
    output = tmp_path / "report.json"
    if case == "wrong_hash_first":
        payload["report_sha256"] = "0" * 64
        payload["unexpected"] = True
        output.write_bytes(_canonical_json_file_bytes(payload))
    elif case == "duplicate_key":
        raw = _canonical_json_file_bytes(payload)
        output.write_bytes(raw.replace(b'{', b'{"schema_version":"1.0.0",', 1))
    elif case == "nested_duplicate_key":
        raw = _canonical_json_file_bytes(payload)
        output.write_bytes(
            raw.replace(
                b'"anchor":{"anchor_text":',
                b'"anchor":{"char_start":55,"anchor_text":',
                1,
            )
        )
    elif case == "nan":
        raw = _canonical_json_file_bytes(payload)
        output.write_bytes(
            raw.replace(
                str(EXPECTED_FIXTURE_SIZE_BYTES).encode("ascii"),
                b"NaN",
                1,
            )
        )
    elif case == "infinity":
        raw = _canonical_json_file_bytes(payload)
        output.write_bytes(
            raw.replace(
                str(EXPECTED_FIXTURE_SIZE_BYTES).encode("ascii"),
                b"Infinity",
                1,
            )
        )
    elif case == "bom":
        output.write_bytes(b"\xef\xbb\xbf" + _canonical_json_file_bytes(payload))
    elif case == "noncanonical_spacing":
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    elif case == "missing_newline":
        output.write_bytes(_canonical_json_file_bytes(payload).removesuffix(b"\n"))
    elif case == "crlf":
        output.write_bytes(_canonical_json_file_bytes(payload).removesuffix(b"\n") + b"\r\n")
    elif case == "extra_newline":
        output.write_bytes(_canonical_json_file_bytes(payload) + b"\n")
    elif case == "coercing_boolean":
        payload["reference_profile_verified"] = "true"
        _rehash_report_payload(payload)
        output.write_bytes(_canonical_json_file_bytes(payload))
    else:
        payload["pdf_size_bytes"] = str(payload["pdf_size_bytes"])
        _rehash_report_payload(payload)
        output.write_bytes(_canonical_json_file_bytes(payload))

    if case == "wrong_hash_first":
        with pytest.raises(ValueError, match="raw canonical report payload"):
            module.load_pdf_anchor_report(output)
    else:
        with pytest.raises(ValueError):
            module.load_pdf_anchor_report(output)


def test_atomic_writer_preserves_old_output_and_cleans_its_temp_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    output = tmp_path / "report.json"
    output.write_bytes(b"existing\n")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        module.write_pdf_anchor_report(output, _create_report(module))

    assert output.read_bytes() == b"existing\n"
    assert tuple(tmp_path.glob(".report.json.*.tmp")) == ()


def test_writer_requires_existing_parent_and_creates_no_directory(tmp_path: Path) -> None:
    module = _pdf_anchor()
    output = tmp_path / "missing" / "report.json"

    with pytest.raises(ValueError, match="parent directory"):
        module.write_pdf_anchor_report(output, _create_report(module))

    assert not output.parent.exists()


@pytest.mark.parametrize("forgery", ["model_copy", "model_construct"])
def test_writer_revalidates_forged_report_before_creating_tempfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forgery: str,
) -> None:
    module = _pdf_anchor()
    valid = _create_report(module)
    if forgery == "model_copy":
        report = valid.model_copy(update={"pdf_size_bytes": 1})
    else:
        report = module.PdfAnchorReport.model_construct(
            **{**valid.__dict__, "pdf_size_bytes": 1}
        )

    def unexpected_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        raise AssertionError("mkstemp must run only after complete validation")

    monkeypatch.setattr(module.tempfile, "mkstemp", unexpected_mkstemp)

    with pytest.raises(ValueError):
        module.write_pdf_anchor_report(tmp_path / "report.json", report)


@pytest.mark.parametrize("operation", ["writer", "replay"])
@pytest.mark.parametrize(
    "forgery",
    [
        "anchor_page_float",
        "box_coordinate_float",
        "profile_float",
        "profile_tuple_list",
        "render_profile_float",
        "render_evidence_float",
        "nested_dict",
    ],
)
def test_writer_and_replay_reject_original_python_normalization_forgeries_before_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    forgery: str,
) -> None:
    module = _pdf_anchor()
    valid = _create_report(module)
    forged = _forge_normalization_preserving_report(module, valid, forgery)
    io_events: list[str] = []

    def unexpected_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        io_events.append("mkstemp")
        raise AssertionError("A forged report must fail before mkstemp.")

    def unexpected_snapshot(source: Path) -> Any:
        io_events.append("source_open")
        raise AssertionError("A forged report must fail before opening the source.")

    monkeypatch.setattr(module.tempfile, "mkstemp", unexpected_mkstemp)
    monkeypatch.setattr(module, "_read_pdf_snapshot", unexpected_snapshot)

    with pytest.raises(module.PdfAnchorOperationalError):
        if operation == "writer":
            module.write_pdf_anchor_report(tmp_path / "report.json", forged)
        else:
            module.verify_pdf_anchor_replay(source=FIXTURE_PATH, report=forged)

    assert io_events == []


def test_legitimate_report_revalidation_still_precedes_writer_and_replay_io(
    tmp_path: Path,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    output = tmp_path / "report.json"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    report = _create_report(module, source=source)

    module.write_pdf_anchor_report(output, report)
    module.verify_pdf_anchor_replay(source=source, report=report)

    assert module.load_pdf_anchor_report(output) == report


@pytest.mark.parametrize(
    "failure_stage",
    ["fdopen", "write", "flush", "fsync", "close", "replace"],
)
def test_writer_cleans_only_its_temp_and_preserves_output_at_every_io_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    module = _pdf_anchor()
    output = tmp_path / "report.json"
    output.write_bytes(b"existing\n")
    real_fdopen = module.os.fdopen
    real_fsync = module.os.fsync
    real_replace = module.os.replace

    class FailingFile:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        @property
        def closed(self) -> bool:
            return self.handle.closed

        def fileno(self) -> int:
            return self.handle.fileno()

        def write(self, data: bytes) -> int:
            if failure_stage == "write":
                raise OSError("simulated write failure")
            return self.handle.write(data)

        def flush(self) -> None:
            if failure_stage == "flush":
                raise OSError("simulated flush failure")
            self.handle.flush()

        def close(self) -> None:
            self.handle.close()
            if failure_stage == "close":
                raise OSError("simulated close failure")

    def failing_fdopen(*args: object, **kwargs: object) -> Any:
        if failure_stage == "fdopen":
            raise OSError("simulated fdopen failure")
        return FailingFile(real_fdopen(*args, **kwargs))

    def failing_fsync(fd: int) -> None:
        if failure_stage == "fsync":
            raise OSError("simulated fsync failure")
        real_fsync(fd)

    def failing_replace(source: Path, destination: Path) -> None:
        if failure_stage == "replace":
            raise OSError("simulated replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "fdopen", failing_fdopen)
    monkeypatch.setattr(module.os, "fsync", failing_fsync)
    monkeypatch.setattr(module.os, "replace", failing_replace)

    with pytest.raises(OSError, match=f"simulated {failure_stage} failure"):
        module.write_pdf_anchor_report(output, _create_report(module))

    assert output.read_bytes() == b"existing\n"
    assert tuple(tmp_path.glob(".report.json.*.tmp")) == ()


def test_writer_rejects_short_write_before_flush_fsync_or_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    output = tmp_path / "report.json"
    output.write_bytes(b"existing\n")
    real_fdopen = module.os.fdopen
    events: list[str] = []

    class ShortWriteFile:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def write(self, data: bytes) -> int:
            events.append("write")
            self.handle.write(data[:-1])
            return len(data) - 1

        def flush(self) -> None:
            events.append("flush")
            raise AssertionError("A short write must fail before flush.")

        def fileno(self) -> int:
            events.append("fileno")
            return self.handle.fileno()

        def close(self) -> None:
            events.append("close")
            self.handle.close()

    def short_fdopen(*args: object, **kwargs: object) -> ShortWriteFile:
        return ShortWriteFile(real_fdopen(*args, **kwargs))

    def unexpected_replace(source: Path, destination: Path) -> None:
        events.append("replace")
        raise AssertionError("A short write must fail before replace.")

    monkeypatch.setattr(module.os, "fdopen", short_fdopen)
    monkeypatch.setattr(module.os, "replace", unexpected_replace)

    with pytest.raises(
        module.PdfAnchorOperationalError,
        match="PDF anchor report write was incomplete",
    ):
        module.write_pdf_anchor_report(output, _create_report(module))

    assert events == ["write", "close"]
    assert output.read_bytes() == b"existing\n"
    assert tuple(tmp_path.glob(".report.json.*.tmp")) == ()


def test_writer_preserves_primary_when_fallback_close_raises_non_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    output = tmp_path / "report.json"
    output.write_bytes(b"existing\n")
    real_fdopen = module.os.fdopen
    events: list[str] = []

    class PrimaryWriteAndSecondaryCloseFailure:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def write(self, data: bytes) -> int:
            events.append("write")
            raise OSError("primary write failure")

        def close(self) -> None:
            events.append("close")
            self.handle.close()
            raise RuntimeError("secondary close failure")

    def failing_fdopen(
        *args: object,
        **kwargs: object,
    ) -> PrimaryWriteAndSecondaryCloseFailure:
        return PrimaryWriteAndSecondaryCloseFailure(real_fdopen(*args, **kwargs))

    monkeypatch.setattr(module.os, "fdopen", failing_fdopen)

    with pytest.raises(OSError, match="primary write failure") as captured:
        module.write_pdf_anchor_report(output, _create_report(module))

    assert "secondary close failure" not in str(captured.value)
    assert events == ["write", "close"]
    assert output.read_bytes() == b"existing\n"
    assert tuple(tmp_path.glob(".report.json.*.tmp")) == ()


def test_writer_retries_first_temp_unlink_failure_and_preserves_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    output = tmp_path / "report.json"
    sentinel = tmp_path / "sentinel.keep"
    output.write_bytes(b"existing\n")
    sentinel.write_bytes(b"keep\n")
    real_unlink = Path.unlink
    unlink_attempts: list[Path] = []

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("primary replace failure")

    def flaky_unlink(path: Path, *, missing_ok: bool = False) -> None:
        unlink_attempts.append(path)
        if len(unlink_attempts) == 1:
            raise OSError("transient unlink failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(module.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    with pytest.raises(OSError, match="primary replace failure"):
        module.write_pdf_anchor_report(output, _create_report(module))

    assert len(unlink_attempts) == 2
    assert unlink_attempts[0] == unlink_attempts[1]
    assert unlink_attempts[0].name.startswith(".report.json.")
    assert output.read_bytes() == b"existing\n"
    assert sentinel.read_bytes() == b"keep\n"
    assert tuple(tmp_path.glob(".report.json.*.tmp")) == ()


def test_writer_annotates_primary_if_temp_deletion_remains_impossible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    output = tmp_path / "report.json"
    sentinel = tmp_path / "sentinel.keep"
    output.write_bytes(b"existing\n")
    sentinel.write_bytes(b"keep\n")
    real_unlink = Path.unlink
    unlink_attempts: list[Path] = []

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("primary replace failure")

    def always_fail_temp_unlink(path: Path, *, missing_ok: bool = False) -> None:
        unlink_attempts.append(path)
        raise OSError("persistent unlink failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", always_fail_temp_unlink)

    with pytest.raises(OSError, match="primary replace failure") as captured:
        module.write_pdf_anchor_report(output, _create_report(module))

    temporary_files = tuple(tmp_path.glob(".report.json.*.tmp"))
    assert len(unlink_attempts) == 2
    assert len(temporary_files) == 1
    assert any(
        "Temporary report cleanup failed" in note
        for note in getattr(captured.value, "__notes__", ())
    )
    assert output.read_bytes() == b"existing\n"
    assert sentinel.read_bytes() == b"keep\n"
    real_unlink(temporary_files[0])


def test_report_and_replay_each_open_and_read_source_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    report = _create_report(module, source=source)
    real_open = Path.open
    events: list[str] = []

    class TrackingHandle:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> "TrackingHandle":
            return self

        def __exit__(self, *args: object) -> None:
            self.handle.close()
            events.append("close")

        def read(self, *args: object) -> bytes:
            events.append("read")
            return self.handle.read(*args)

    def tracking_open(path: Path, *args: object, **kwargs: object) -> Any:
        handle = real_open(path, *args, **kwargs)
        if path.resolve() == source.resolve():
            events.append("open")
            return TrackingHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)

    _create_report(module, source=source)
    assert events == ["open", "read", "close"]
    events.clear()
    module.verify_pdf_anchor_replay(source=source, report=report)
    assert events == ["open", "read", "close"]


@pytest.mark.parametrize("operation", ["report", "replay"])
def test_report_and_replay_share_snapshot_when_path_mutates_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    report = _create_report(module, source=source)
    real_locate = module._locate_snapshot_with_page_count
    mutations = 0

    def mutate_then_locate(*args: object, **kwargs: object) -> Any:
        nonlocal mutations
        mutations += 1
        source.write_bytes(b"%PDF-mutated-after-snapshot")
        return real_locate(*args, **kwargs)

    monkeypatch.setattr(module, "_locate_snapshot_with_page_count", mutate_then_locate)

    if operation == "report":
        observed = _create_report(module, source=source)
        assert observed.pdf_sha256 == EXPECTED_FIXTURE_SHA256
    else:
        module.verify_pdf_anchor_replay(source=source, report=report)

    assert mutations == 1
    assert source.read_bytes() == b"%PDF-mutated-after-snapshot"


def test_write_less_replay_succeeds_and_creates_no_artifacts(tmp_path: Path) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    report = _create_report(module, source=source)
    before = tuple(sorted(path.name for path in tmp_path.iterdir()))

    result = module.verify_pdf_anchor_replay(source=source, report=report)

    assert result is None
    assert tuple(sorted(path.name for path in tmp_path.iterdir())) == before


@pytest.mark.parametrize("payload_kind", ["rgba", "png"])
def test_report_factory_rejects_render_bytes_that_disagree_with_evidence(
    monkeypatch: pytest.MonkeyPatch,
    payload_kind: str,
) -> None:
    module = _pdf_anchor()
    real_render = module._render_pdf_page_snapshot

    def mismatched_render(*args: object, **kwargs: object) -> Any:
        rendered = real_render(*args, **kwargs)
        return rendered.__class__(
            physical_page_index=rendered.physical_page_index,
            page_width_points=rendered.page_width_points,
            page_height_points=rendered.page_height_points,
            pixel_width=rendered.pixel_width,
            pixel_height=rendered.pixel_height,
            packed_rgba_bytes=(
                rendered.packed_rgba_bytes + b"x"
                if payload_kind == "rgba"
                else rendered.packed_rgba_bytes
            ),
            png_bytes=(
                rendered.png_bytes + b"x"
                if payload_kind == "png"
                else rendered.png_bytes
            ),
            evidence=rendered.evidence,
        )

    monkeypatch.setattr(module, "_render_pdf_page_snapshot", mismatched_render)

    with pytest.raises(ValueError, match="render evidence"):
        _create_report(module)


@pytest.mark.parametrize(
    "mismatch",
    [
        "profile",
        "tool",
        "page_count",
        "rotation",
        "binding",
        "offsets",
        "canonical_text",
        "boxes",
        "evidence_id",
        "rgba",
        "png",
        "forged_model",
    ],
)
def test_replay_rejects_every_stored_or_fresh_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    report = _create_report(module, source=source)

    if mismatch in {"profile", "tool", "forged_model"}:
        payload = report.model_dump(mode="json")
        if mismatch == "profile":
            field = "profile_sha256"
            value: object = "0" * 64
        elif mismatch == "tool":
            field = "python_version"
            value = "0.0.0"
        else:
            field = "pdf_size_bytes"
            value = EXPECTED_FIXTURE_SIZE_BYTES + 1
        payload[field] = value
        _rehash_report_payload(payload)
        updates = {field: value, "report_sha256": payload["report_sha256"]}
        if mismatch == "forged_model":
            forged = module.PdfAnchorReport.model_construct(
                **{**report.__dict__, **updates}
            )
        else:
            forged = report.model_copy(update=updates)
        with pytest.raises(ValueError):
            module.verify_pdf_anchor_replay(source=source, report=forged)
        return

    real_locate = module._locate_snapshot_with_page_count
    if mismatch == "page_count":
        def wrong_page_count(*args: object, **kwargs: object) -> tuple[Any, int]:
            anchor, page_count = real_locate(*args, **kwargs)
            return anchor, page_count + 1

        monkeypatch.setattr(
            module,
            "_locate_snapshot_with_page_count",
            wrong_page_count,
        )
    elif mismatch in {
        "rotation",
        "binding",
        "offsets",
        "canonical_text",
        "boxes",
        "evidence_id",
    }:
        def wrong_anchor(*args: object, **kwargs: object) -> tuple[Any, int]:
            anchor, page_count = real_locate(*args, **kwargs)
            assert anchor is not None
            updates: dict[str, object]
            if mismatch == "rotation":
                updates = {"source_page_rotation_degrees": 90}
            elif mismatch == "binding":
                updates = {"file_version_binding_sha256": "0" * 64}
            elif mismatch == "offsets":
                updates = {"char_start": anchor.char_start - 1}
            elif mismatch == "canonical_text":
                updates = {"canonical_page_text": "tampered"}
            elif mismatch == "boxes":
                updates = {"boxes": ()}
            else:
                updates = {"evidence_id": "ev-sha256-" + "0" * 64}
            return anchor.model_copy(update=updates), page_count

        monkeypatch.setattr(module, "_locate_snapshot_with_page_count", wrong_anchor)
    else:
        real_render = module._render_pdf_page_snapshot

        def wrong_render(*args: object, **kwargs: object) -> Any:
            rendered = real_render(*args, **kwargs)
            field = "rgba_sha256" if mismatch == "rgba" else "png_sha256"
            evidence = rendered.evidence.model_copy(update={field: "0" * 64})
            return rendered.__class__(
                physical_page_index=rendered.physical_page_index,
                page_width_points=rendered.page_width_points,
                page_height_points=rendered.page_height_points,
                pixel_width=rendered.pixel_width,
                pixel_height=rendered.pixel_height,
                packed_rgba_bytes=rendered.packed_rgba_bytes,
                png_bytes=rendered.png_bytes,
                evidence=evidence,
            )

        monkeypatch.setattr(module, "_render_pdf_page_snapshot", wrong_render)

    with pytest.raises(ValueError, match="replay"):
        module.verify_pdf_anchor_replay(source=source, report=report)


def test_replay_rejects_wrong_source_anchor_and_render_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    report = _create_report(module, source=source)

    source.write_bytes(FIXTURE_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="replay source"):
        module.verify_pdf_anchor_replay(source=source, report=report)
    source.write_bytes(FIXTURE_PATH.read_bytes())

    real_locate = module._locate_snapshot_with_page_count

    def wrong_anchor(*args: object, **kwargs: object) -> tuple[Any, int]:
        anchor, page_count = real_locate(*args, **kwargs)
        assert anchor is not None
        return anchor.model_copy(update={"printed_page_label": "A-8"}), page_count

    monkeypatch.setattr(module, "_locate_snapshot_with_page_count", wrong_anchor)
    with pytest.raises(ValueError, match="replay text anchor"):
        module.verify_pdf_anchor_replay(source=source, report=report)

    monkeypatch.setattr(module, "_locate_snapshot_with_page_count", real_locate)
    real_render = module._render_pdf_page_snapshot

    def wrong_render(*args: object, **kwargs: object) -> Any:
        rendered = real_render(*args, **kwargs)
        evidence = rendered.evidence.model_copy(update={"png_sha256": "0" * 64})
        return rendered.__class__(
            physical_page_index=rendered.physical_page_index,
            page_width_points=rendered.page_width_points,
            page_height_points=rendered.page_height_points,
            pixel_width=rendered.pixel_width,
            pixel_height=rendered.pixel_height,
            packed_rgba_bytes=rendered.packed_rgba_bytes,
            png_bytes=rendered.png_bytes,
            evidence=evidence,
        )

    monkeypatch.setattr(module, "_render_pdf_page_snapshot", wrong_render)
    with pytest.raises(ValueError, match="replay render"):
        module.verify_pdf_anchor_replay(source=source, report=report)


def test_report_and_replay_close_pdf_before_immediate_windows_rename_delete(
    tmp_path: Path,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    moved = tmp_path / "moved.pdf"
    source.write_bytes(FIXTURE_PATH.read_bytes())

    report = _create_report(module, source=source)
    os.replace(source, moved)
    os.replace(moved, source)
    module.verify_pdf_anchor_replay(source=source, report=report)
    os.replace(source, moved)
    moved.unlink()

    assert not source.exists()
    assert not moved.exists()


def _success_cli_arguments(*, source: Path, hardware: Path, output: Path) -> list[str]:
    return [
        "--pdf",
        str(source),
        "--file-version-id",
        FILE_VERSION_ID,
        "--needle",
        ANCHOR_SENTENCE,
        "--hardware-facts",
        str(hardware),
        "--output",
        str(output),
    ]


def test_cli_success_is_silent_and_writes_strict_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _pdf_anchor()
    output = tmp_path / "report.json"

    result = module.main(
        _success_cli_arguments(
            source=FIXTURE_PATH,
            hardware=HARDWARE_FACTS_PATH,
            output=output,
        )
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == ""
    assert captured.err == ""
    assert module.load_pdf_anchor_report(output).pdf_sha256 == EXPECTED_FIXTURE_SHA256


def test_cli_missing_arguments_return_two_with_one_stable_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _pdf_anchor()

    result = module.main([])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == "Invalid command arguments.\n"


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("no_match", "The normalized anchor text was not found."),
        ("invalid_pdf", "The source file is not a PDF."),
        ("invalid_hardware", "Hardware facts file is not canonical."),
        ("missing_parent", "Output parent directory does not exist."),
        ("output_pdf_alias", "Output path must not alias an input path."),
        ("output_hardware_alias", "Output path must not alias an input path."),
    ],
)
def test_cli_operational_failures_are_silent_on_stdout_and_leave_no_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    case: str,
    expected_error: str,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    hardware = tmp_path / "hardware.json"
    hardware.write_bytes(HARDWARE_FACTS_PATH.read_bytes())
    output = tmp_path / "report.json"
    needle = ANCHOR_SENTENCE
    if case == "no_match":
        needle = "This exact sentence is absent."
    elif case == "invalid_pdf":
        source.write_bytes(b"not a PDF")
    elif case == "invalid_hardware":
        hardware.write_bytes(b"{}\n")
    elif case == "missing_parent":
        output = tmp_path / "missing" / "report.json"
    elif case == "output_pdf_alias":
        output = Path(str(source).upper())
    elif case == "output_hardware_alias":
        output = Path(str(hardware).upper())

    arguments = _success_cli_arguments(source=source, hardware=hardware, output=output)
    arguments[arguments.index(ANCHOR_SENTENCE)] = needle
    result = module.main(arguments)

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == expected_error + "\n"
    if case == "missing_parent":
        assert not output.parent.exists()
    elif case not in {"output_pdf_alias", "output_hardware_alias"}:
        assert not output.exists()
    assert tuple(tmp_path.glob(".*.tmp")) == ()


def test_cli_writer_runs_after_pdf_handles_close_and_preserves_old_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    moved = tmp_path / "moved.pdf"
    output = tmp_path / "report.json"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    output.write_bytes(b"existing\n")
    real_writer = module.write_pdf_anchor_report

    def assert_closed_then_fail(path: Path, report: object) -> None:
        os.replace(source, moved)
        os.replace(moved, source)
        raise OSError("simulated writer failure")

    monkeypatch.setattr(module, "write_pdf_anchor_report", assert_closed_then_fail)

    result = module.main(
        _success_cli_arguments(
            source=source,
            hardware=HARDWARE_FACTS_PATH,
            output=output,
        )
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "PDF anchor operation failed.\n"
    assert output.read_bytes() == b"existing\n"
    assert tuple(tmp_path.glob(".*.tmp")) == ()
    monkeypatch.setattr(module, "write_pdf_anchor_report", real_writer)


def test_cli_short_write_returns_stable_domain_error_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    output = tmp_path / "report.json"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    output.write_bytes(b"existing\n")
    real_fdopen = module.os.fdopen
    events: list[str] = []

    class ShortWriteFile:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def write(self, data: bytes) -> int:
            events.append("write")
            self.handle.write(data[:-1])
            return len(data) - 1

        def flush(self) -> None:
            events.append("flush")
            raise AssertionError("A short write must fail before flush.")

        def close(self) -> None:
            events.append("close")
            self.handle.close()

    def short_fdopen(*args: object, **kwargs: object) -> ShortWriteFile:
        return ShortWriteFile(real_fdopen(*args, **kwargs))

    monkeypatch.setattr(module.os, "fdopen", short_fdopen)

    result = module.main(
        _success_cli_arguments(
            source=source,
            hardware=HARDWARE_FACTS_PATH,
            output=output,
        )
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "PDF anchor report write was incomplete.\n"
    assert events == ["write", "close"]
    assert output.read_bytes() == b"existing\n"
    assert tuple(tmp_path.glob(".*.tmp")) == ()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--unknown"],
        ["unexpected-positional"],
        ["--pdf", "only.pdf", "--extra", "value"],
    ],
)
def test_cli_unknown_or_extra_arguments_return_exact_argument_error(
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    module = _pdf_anchor()

    result = module.main(arguments)

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert captured.err == "Invalid command arguments.\n"


def test_cli_rejects_relative_dotdot_output_alias_before_any_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    hardware = tmp_path / "hardware.json"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    hardware.write_bytes(HARDWARE_FACTS_PATH.read_bytes())
    monkeypatch.chdir(tmp_path)

    def unexpected_factory(*args: object, **kwargs: object) -> Any:
        raise AssertionError("Aliased paths must be rejected before input reads.")

    monkeypatch.setattr(module, "create_pdf_anchor_report", unexpected_factory)

    result = module.main(
        _success_cli_arguments(
            source=Path("source.pdf"),
            hardware=Path("hardware.json"),
            output=Path("child") / ".." / "source.pdf",
        )
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "Output path must not alias an input path.\n"


@pytest.mark.parametrize("failure", ["no_match", "duplicate", "render", "hardware"])
def test_cli_preserves_existing_output_for_prepublication_operational_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    hardware = tmp_path / "hardware.json"
    output = tmp_path / "report.json"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    hardware.write_bytes(HARDWARE_FACTS_PATH.read_bytes())
    output.write_bytes(b"existing\n")
    needle = ANCHOR_SENTENCE
    expected_error = "The normalized anchor text was not found."
    if failure == "no_match":
        needle = "This exact sentence is absent."
    elif failure == "duplicate":
        def duplicate_match(*args: object, **kwargs: object) -> Any:
            raise module.AmbiguousAnchorError(
                "The normalized anchor text matched more than once."
            )

        monkeypatch.setattr(
            module,
            "_locate_snapshot_with_page_count",
            duplicate_match,
        )
        expected_error = "The normalized anchor text matched more than once."
    elif failure == "hardware":
        hardware.write_bytes(b"{}\n")
        expected_error = "Hardware facts file is not canonical."
    else:
        def fail_render(*args: object, **kwargs: object) -> Any:
            raise module.PdfAnchorOperationalError("simulated render failure")

        monkeypatch.setattr(module, "_render_pdf_page_snapshot", fail_render)
        expected_error = "simulated render failure"

    arguments = _success_cli_arguments(
        source=source,
        hardware=hardware,
        output=output,
    )
    arguments[arguments.index(ANCHOR_SENTENCE)] = needle
    result = module.main(arguments)

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == expected_error + "\n"
    assert output.read_bytes() == b"existing\n"
    assert tuple(tmp_path.glob(".*.tmp")) == ()


@pytest.mark.parametrize(
    "needle",
    [
        ANCHOR_SENTENCE.replace("The anchor", "The\u00a0anchor"),
        ANCHOR_SENTENCE.replace("accuracy of", "accuracy\u2003of"),
    ],
)
def test_cli_normalization_equivalent_raw_needle_preserves_old_output_and_no_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    needle: str,
) -> None:
    module = _pdf_anchor()
    output = tmp_path / "report.json"
    output.write_bytes(b"existing\n")

    def unexpected_render(*args: object, **kwargs: object) -> Any:
        raise AssertionError("A non-default raw needle must fail before rendering.")

    monkeypatch.setattr(module, "_render_pdf_page_snapshot", unexpected_render)
    arguments = _success_cli_arguments(
        source=FIXTURE_PATH,
        hardware=HARDWARE_FACTS_PATH,
        output=output,
    )
    arguments[arguments.index(ANCHOR_SENTENCE)] = needle

    result = module.main(arguments)

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == (
        "The requested anchor text does not exactly match the reference profile.\n"
    )
    assert output.read_bytes() == b"existing\n"
    assert tuple(tmp_path.glob(".*.tmp")) == ()


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_cli_does_not_swallow_arbitrary_programming_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
) -> None:
    module = _pdf_anchor()
    output = tmp_path / "report.json"

    def fail_unexpectedly(*args: object, **kwargs: object) -> Any:
        raise error_type("unexpected implementation defect")

    monkeypatch.setattr(module, "_render_pdf_page_snapshot", fail_unexpectedly)

    with pytest.raises(error_type, match="unexpected implementation defect"):
        module.main(
            _success_cli_arguments(
                source=FIXTURE_PATH,
                hardware=HARDWARE_FACTS_PATH,
                output=output,
            )
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert not output.exists()


def test_cli_translates_expected_injected_render_domain_failure_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _pdf_anchor()
    output = tmp_path / "report.json"

    def fail_render(*args: object, **kwargs: object) -> Any:
        raise module.PdfAnchorOperationalError("PDF anchor render failed.")

    monkeypatch.setattr(module, "_render_pdf_page_snapshot", fail_render)

    result = module.main(
        _success_cli_arguments(
            source=FIXTURE_PATH,
            hardware=HARDWARE_FACTS_PATH,
            output=output,
        )
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "PDF anchor render failed.\n"
    assert not output.exists()


def test_cli_wraps_path_resolution_os_text_in_stable_domain_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _pdf_anchor()

    def fail_resolve(path: Path, *, strict: bool = False) -> Path:
        raise OSError("secret platform-specific path text")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    result = module.main(
        _success_cli_arguments(
            source=FIXTURE_PATH,
            hardware=HARDWARE_FACTS_PATH,
            output=tmp_path / "report.json",
        )
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "Input/output paths could not be resolved.\n"
    assert "secret" not in captured.err


@pytest.mark.parametrize("alias_input", ["source", "hardware"])
def test_cli_rejects_existing_hardlink_alias_before_input_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    alias_input: str,
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    hardware = tmp_path / "hardware.json"
    output = tmp_path / "output.json"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    hardware.write_bytes(HARDWARE_FACTS_PATH.read_bytes())
    aliased = source if alias_input == "source" else hardware
    os.link(aliased, output)
    original = aliased.read_bytes()

    def unexpected_factory(*args: object, **kwargs: object) -> Any:
        raise AssertionError("Hardlink aliases must fail before input reads.")

    monkeypatch.setattr(module, "create_pdf_anchor_report", unexpected_factory)

    result = module.main(
        _success_cli_arguments(source=source, hardware=hardware, output=output)
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "Output path must not alias an input path.\n"
    assert output.read_bytes() == original
    assert aliased.read_bytes() == original
    assert os.path.samefile(output, aliased)
    assert tuple(tmp_path.glob(".*.tmp")) == ()


def test_cli_translates_samefile_failure_before_input_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _pdf_anchor()
    source = tmp_path / "source.pdf"
    hardware = tmp_path / "hardware.json"
    output = tmp_path / "output.json"
    source.write_bytes(FIXTURE_PATH.read_bytes())
    hardware.write_bytes(HARDWARE_FACTS_PATH.read_bytes())
    output.write_bytes(b"existing\n")

    def fail_samefile(first: Path, second: Path) -> bool:
        raise OSError("secret samefile detail")

    def unexpected_factory(*args: object, **kwargs: object) -> Any:
        raise AssertionError("samefile errors must fail before input reads.")

    monkeypatch.setattr(module.os.path, "samefile", fail_samefile)
    monkeypatch.setattr(module, "create_pdf_anchor_report", unexpected_factory)

    result = module.main(
        _success_cli_arguments(source=source, hardware=hardware, output=output)
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "Input/output path identity could not be checked.\n"
    assert "secret" not in captured.err
    assert output.read_bytes() == b"existing\n"
