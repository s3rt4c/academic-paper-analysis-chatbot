from __future__ import annotations

import copy

import pytest


@pytest.fixture
def frozen_profile_payload() -> dict[str, object]:
    return {
        "schema_version": "embedding-profile-v1",
        "model_repository": "BAAI/bge-small-en-v1.5",
        "model_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "artifacts": [
            {
                "filename": "1_Pooling/config.json",
                "byte_size": 190,
                "sha256": "d1caf60c96f5fba2157c0c26b76d80818fad6cf0b8eb5e73ec372ff9818eba5c",
            },
            {
                "filename": "config.json",
                "byte_size": 743,
                "sha256": "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
            },
            {
                "filename": "config_sentence_transformers.json",
                "byte_size": 124,
                "sha256": "940d5f50db195fa6e5e6a4f122c095f77880de259d74b14a65779ed48bdd7c56",
            },
            {
                "filename": "modules.json",
                "byte_size": 349,
                "sha256": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
            },
            {
                "filename": "onnx/model.onnx",
                "byte_size": 133_093_490,
                "sha256": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
            },
            {
                "filename": "sentence_bert_config.json",
                "byte_size": 52,
                "sha256": "84e39fda68ccbff05bfa723ae9c0e70e23e2ec373b76e0f8c6e71af72a693cbf",
            },
            {
                "filename": "special_tokens_map.json",
                "byte_size": 125,
                "sha256": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
            },
            {
                "filename": "tokenizer.json",
                "byte_size": 711_396,
                "sha256": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
            },
            {
                "filename": "tokenizer_config.json",
                "byte_size": 366,
                "sha256": "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
            },
            {
                "filename": "vocab.txt",
                "byte_size": 231_508,
                "sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
            },
        ],
        "artifact_set_sha256": "adf3c9b45097fa689815068b872abdd279849e423c7281226b91b359297ece64",
        "dimension": 384,
        "max_sequence_length": 512,
        "document_prefix_utf8": "",
        "query_prefix_utf8": "Represent this sentence for searching relevant passages: ",
        "special_token_policy": {
            "add_special_tokens": True,
            "document_content_token_budget": 510,
            "query_prefix_token_count": 8,
            "query_source_token_budget": 502,
            "single_sequence_template": "[CLS] $A [SEP]",
            "special_token_count": 2,
        },
        "pooling": "last_hidden_state[:,0,:]",
        "normalization": {"dtype": "float32", "rule": "l2"},
        "span_policy": "canonical-word-greedy-v1-zero-overlap",
        "tokenizer": {
            "artifact": "tokenizer.json",
            "runtime": "tokenizers==0.20.0",
            "type": "BertTokenizer/WordPiece",
        },
        "onnx_runtime": {
            "provider": "CPUExecutionProvider",
            "requirement": "onnxruntime==1.27.0",
        },
        "onnx_graph": {
            "opset": 11,
            "inputs": [
                ["input_ids", "int64"],
                ["attention_mask", "int64"],
                ["token_type_ids", "int64"],
            ],
            "output": ["last_hidden_state", "float32", "batch_size", "sequence_length", 384],
        },
        "onnx_ir_version": 6,
    }


@pytest.fixture
def mutable_frozen_profile_payload(frozen_profile_payload: dict[str, object]) -> dict[str, object]:
    return copy.deepcopy(frozen_profile_payload)
