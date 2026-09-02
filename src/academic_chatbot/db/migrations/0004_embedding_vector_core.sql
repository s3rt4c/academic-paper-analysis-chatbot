CREATE TABLE embedding_profiles (
    embedding_profile_id TEXT PRIMARY KEY,
    canonical_profile_json TEXT NOT NULL CHECK(canonical_profile_json <> ''),
    canonical_profile_sha256 TEXT NOT NULL CHECK(length(canonical_profile_sha256) = 64),
    artifact_manifest_sha256 TEXT NOT NULL CHECK(length(artifact_manifest_sha256) = 64),
    dimension INTEGER NOT NULL CHECK(dimension > 0),
    span_policy_id TEXT NOT NULL CHECK(span_policy_id <> ''),
    created_at TEXT NOT NULL,
    UNIQUE(canonical_profile_sha256)
) STRICT;

CREATE TRIGGER embedding_profiles_immutable
BEFORE UPDATE ON embedding_profiles
BEGIN
    SELECT RAISE(ABORT, 'embedding profiles are immutable');
END;

CREATE TABLE embedding_spans (
    embedding_span_id TEXT PRIMARY KEY,
    embedding_profile_id TEXT NOT NULL REFERENCES embedding_profiles(embedding_profile_id)
        ON DELETE RESTRICT,
    document_generation_id TEXT NOT NULL
        REFERENCES document_generations(document_generation_id) ON DELETE RESTRICT,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE RESTRICT,
    page_id TEXT NOT NULL REFERENCES pages(page_id) ON DELETE RESTRICT,
    start_offset INTEGER NOT NULL CHECK(start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK(end_offset > start_offset),
    coverage_status TEXT NOT NULL CHECK(coverage_status IN ('EMBEDDABLE', 'EXCLUDED_UNEMBEDDABLE')),
    UNIQUE(embedding_profile_id, document_generation_id, chunk_id, page_id, start_offset, end_offset)
) STRICT;

CREATE TRIGGER embedding_spans_lineage_insert
BEFORE INSERT ON embedding_spans
WHEN NOT EXISTS (
    SELECT 1 FROM chunks AS c JOIN pages AS p ON p.page_id = c.page_id
    WHERE c.chunk_id = NEW.chunk_id
      AND c.page_id = NEW.page_id
      AND c.document_generation_id = NEW.document_generation_id
      AND p.document_generation_id = NEW.document_generation_id
      AND NEW.start_offset >= c.start_offset
      AND NEW.end_offset <= c.end_offset
      AND NEW.end_offset <= length(p.canonical_text)
)
BEGIN
    SELECT RAISE(ABORT, 'embedding span does not match chunk/page/generation lineage');
END;

CREATE TRIGGER embedding_spans_immutable
BEFORE UPDATE ON embedding_spans
BEGIN
    SELECT RAISE(ABORT, 'embedding spans are immutable');
END;

CREATE INDEX embedding_spans_profile_generation_chunk_index
    ON embedding_spans(embedding_profile_id, document_generation_id, chunk_id);

CREATE TABLE vector_generations (
    vector_generation_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE RESTRICT,
    embedding_profile_id TEXT NOT NULL REFERENCES embedding_profiles(embedding_profile_id)
        ON DELETE RESTRICT,
    source_snapshot_sha256 TEXT NOT NULL CHECK(length(source_snapshot_sha256) = 64),
    artifact_relative_dir TEXT NOT NULL CHECK(
        artifact_relative_dir <> ''
        AND substr(artifact_relative_dir, 1, 1) <> '/'
        AND substr(artifact_relative_dir, -1, 1) <> '/'
        AND instr(artifact_relative_dir, char(92)) = 0
        AND instr(artifact_relative_dir, ':') = 0
        AND instr(artifact_relative_dir, '//') = 0
        AND artifact_relative_dir NOT GLOB './*'
        AND artifact_relative_dir NOT GLOB '../*'
        AND artifact_relative_dir NOT GLOB '*/./*'
        AND artifact_relative_dir NOT GLOB '*/../*'
    ),
    state TEXT NOT NULL CHECK(state IN ('DB_CANDIDATE', 'FILES_FINALIZED', 'STALE')),
    vector_store_manifest_sha256 TEXT CHECK(
        vector_store_manifest_sha256 IS NULL OR length(vector_store_manifest_sha256) = 64
    ),
    eligible_native_chunks INTEGER NOT NULL CHECK(eligible_native_chunks >= 0),
    embeddable_spans INTEGER NOT NULL CHECK(embeddable_spans >= 0),
    excluded_unembeddable_spans INTEGER NOT NULL CHECK(excluded_unembeddable_spans >= 0),
    needs_ocr_pages INTEGER NOT NULL CHECK(needs_ocr_pages >= 0),
    indexed_documents INTEGER NOT NULL CHECK(indexed_documents >= 0),
    unindexed_documents INTEGER NOT NULL CHECK(unindexed_documents >= 0),
    created_at TEXT NOT NULL,
    UNIQUE(project_id, embedding_profile_id, source_snapshot_sha256),
    CHECK(
        (state = 'DB_CANDIDATE' AND vector_store_manifest_sha256 IS NULL)
        OR (state = 'FILES_FINALIZED' AND vector_store_manifest_sha256 IS NOT NULL)
        OR state = 'STALE'
    )
) STRICT;

CREATE TRIGGER vector_generations_immutable_identity
BEFORE UPDATE OF vector_generation_id, project_id, embedding_profile_id, source_snapshot_sha256,
                 artifact_relative_dir, eligible_native_chunks, embeddable_spans,
                 excluded_unembeddable_spans, needs_ocr_pages, indexed_documents,
                 unindexed_documents, created_at ON vector_generations
BEGIN
    SELECT RAISE(ABORT, 'vector generation identity and coverage are immutable');
END;

CREATE TRIGGER vector_generations_one_way_state
BEFORE UPDATE OF state, vector_store_manifest_sha256 ON vector_generations
WHEN NOT (
    (OLD.state = 'DB_CANDIDATE' AND NEW.state = 'FILES_FINALIZED'
     AND OLD.vector_store_manifest_sha256 IS NULL AND NEW.vector_store_manifest_sha256 IS NOT NULL)
    OR (OLD.state = 'DB_CANDIDATE' AND NEW.state = 'STALE'
        AND OLD.vector_store_manifest_sha256 IS NULL AND NEW.vector_store_manifest_sha256 IS NULL)
    OR (OLD.state = 'FILES_FINALIZED' AND NEW.state = 'STALE'
        AND OLD.vector_store_manifest_sha256 IS NEW.vector_store_manifest_sha256)
)
BEGIN
    SELECT RAISE(ABORT, 'vector generation state transition is not allowed');
END;

CREATE TRIGGER vector_generations_published_finalized
BEFORE UPDATE OF state, vector_store_manifest_sha256 ON vector_generations
WHEN EXISTS (
    SELECT 1 FROM vector_generation_publications
    WHERE vector_generation_id = OLD.vector_generation_id
)
BEGIN
    SELECT RAISE(ABORT, 'published vector generation metadata is immutable');
END;

CREATE TABLE vector_generation_sources (
    vector_generation_id TEXT NOT NULL REFERENCES vector_generations(vector_generation_id)
        ON DELETE RESTRICT,
    file_version_id TEXT NOT NULL REFERENCES file_versions(file_version_id) ON DELETE RESTRICT,
    document_generation_id TEXT NOT NULL
        REFERENCES document_generations(document_generation_id) ON DELETE RESTRICT,
    eligible_native_chunk_count INTEGER NOT NULL CHECK(eligible_native_chunk_count >= 0),
    needs_ocr_page_count INTEGER NOT NULL CHECK(needs_ocr_page_count >= 0),
    PRIMARY KEY(vector_generation_id, file_version_id),
    UNIQUE(vector_generation_id, document_generation_id)
) STRICT;

CREATE TRIGGER vector_generation_sources_lineage_insert
BEFORE INSERT ON vector_generation_sources
WHEN NOT EXISTS (
    SELECT 1 FROM vector_generations AS vg
    JOIN file_versions AS fv ON fv.file_version_id = NEW.file_version_id
    JOIN papers AS paper ON paper.paper_id = fv.paper_id
    JOIN document_generations AS dg ON dg.document_generation_id = NEW.document_generation_id
    JOIN generation_publications AS active
      ON active.file_version_id = NEW.file_version_id
     AND active.document_generation_id = NEW.document_generation_id
    WHERE vg.vector_generation_id = NEW.vector_generation_id
      AND paper.project_id = vg.project_id
      AND dg.file_version_id = NEW.file_version_id
)
BEGIN
    SELECT RAISE(ABORT, 'vector generation source does not belong to project/file version');
END;

CREATE TRIGGER vector_generation_sources_immutable
BEFORE UPDATE ON vector_generation_sources
BEGIN
    SELECT RAISE(ABORT, 'vector generation sources are immutable');
END;

CREATE TABLE vector_generation_spans (
    vector_generation_id TEXT NOT NULL REFERENCES vector_generations(vector_generation_id)
        ON DELETE RESTRICT,
    vector_row INTEGER NOT NULL CHECK(vector_row >= 0),
    embedding_span_id TEXT NOT NULL REFERENCES embedding_spans(embedding_span_id)
        ON DELETE RESTRICT,
    PRIMARY KEY(vector_generation_id, vector_row),
    UNIQUE(vector_generation_id, embedding_span_id)
) STRICT;

CREATE TRIGGER vector_generation_spans_lineage_insert
BEFORE INSERT ON vector_generation_spans
WHEN NOT EXISTS (
    SELECT 1 FROM vector_generations AS vg
    JOIN embedding_spans AS span ON span.embedding_span_id = NEW.embedding_span_id
    JOIN vector_generation_sources AS source
      ON source.vector_generation_id = vg.vector_generation_id
     AND source.document_generation_id = span.document_generation_id
    WHERE vg.vector_generation_id = NEW.vector_generation_id
      AND span.embedding_profile_id = vg.embedding_profile_id
      AND span.coverage_status = 'EMBEDDABLE'
)
BEGIN
    SELECT RAISE(ABORT, 'vector row span must be embeddable and belong to the generation source/profile');
END;

CREATE TRIGGER vector_generation_spans_candidate_only_insert
BEFORE INSERT ON vector_generation_spans
WHEN NOT EXISTS (
    SELECT 1 FROM vector_generations
    WHERE vector_generation_id = NEW.vector_generation_id AND state = 'DB_CANDIDATE'
)
BEGIN
    SELECT RAISE(ABORT, 'only candidate vector generations may change row mappings');
END;

CREATE TRIGGER vector_generation_spans_immutable_update
BEFORE UPDATE ON vector_generation_spans
BEGIN
    SELECT RAISE(ABORT, 'vector row mappings are immutable');
END;

CREATE TRIGGER vector_generation_spans_candidate_only_delete
BEFORE DELETE ON vector_generation_spans
WHEN NOT EXISTS (
    SELECT 1 FROM vector_generations
    WHERE vector_generation_id = OLD.vector_generation_id AND state = 'DB_CANDIDATE'
)
BEGIN
    SELECT RAISE(ABORT, 'only candidate vector generations may change row mappings');
END;

CREATE INDEX vector_generation_spans_span_index
    ON vector_generation_spans(embedding_span_id);

CREATE TABLE vector_generation_publications (
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE RESTRICT,
    embedding_profile_id TEXT NOT NULL REFERENCES embedding_profiles(embedding_profile_id)
        ON DELETE RESTRICT,
    vector_generation_id TEXT NOT NULL UNIQUE
        REFERENCES vector_generations(vector_generation_id) ON DELETE RESTRICT,
    PRIMARY KEY(project_id, embedding_profile_id)
) STRICT;

CREATE TRIGGER vector_generation_publications_finalized_insert
BEFORE INSERT ON vector_generation_publications
WHEN NOT EXISTS (
    SELECT 1 FROM vector_generations
    WHERE vector_generation_id = NEW.vector_generation_id
      AND project_id = NEW.project_id
      AND embedding_profile_id = NEW.embedding_profile_id
      AND state = 'FILES_FINALIZED'
)
BEGIN
    SELECT RAISE(ABORT, 'publication must reference a finalized matching vector generation');
END;

CREATE TRIGGER vector_generation_publications_finalized_update
BEFORE UPDATE OF project_id, embedding_profile_id, vector_generation_id ON vector_generation_publications
WHEN NOT EXISTS (
    SELECT 1 FROM vector_generations
    WHERE vector_generation_id = NEW.vector_generation_id
      AND project_id = NEW.project_id
      AND embedding_profile_id = NEW.embedding_profile_id
      AND state = 'FILES_FINALIZED'
)
BEGIN
    SELECT RAISE(ABORT, 'publication must reference a finalized matching vector generation');
END;
