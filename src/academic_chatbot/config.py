from pydantic import BaseModel, Field


class RuntimeProfile(BaseModel, frozen=True):
    profile_id: str = "local-q4-ctx4k-v1"
    context_tokens: int = 4096
    max_evidence_tokens: int = 2000
    min_evidence_excerpt_tokens: int = 180
    max_evidence_excerpt_tokens: int = 320
    max_output_tokens: int = 1024
    max_heavy_jobs: int = 1
    peak_process_tree_mib: int = Field(default=12 * 1024, gt=0)
