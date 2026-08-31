CREATE TABLE file_versions_new (
    file_version_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE RESTRICT,
    sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
    original_relative_path TEXT NOT NULL CHECK(
        original_relative_path <> ''
        AND substr(original_relative_path, 1, 1) <> '/'
        AND substr(original_relative_path, -1, 1) <> '/'
        AND instr(original_relative_path, char(92)) = 0
        AND instr(original_relative_path, ':') = 0
        AND instr(original_relative_path, '//') = 0
        AND original_relative_path <> '.'
        AND original_relative_path <> '..'
        AND original_relative_path NOT GLOB './*'
        AND original_relative_path NOT GLOB '../*'
        AND original_relative_path NOT GLOB '*/.'
        AND original_relative_path NOT GLOB '*/..'
        AND original_relative_path NOT GLOB '*/./*'
        AND original_relative_path NOT GLOB '*/../*'
    ),
    created_at TEXT NOT NULL,
    UNIQUE(paper_id, sha256)
) STRICT;

INSERT INTO file_versions_new
    (file_version_id, paper_id, sha256, original_relative_path, created_at)
SELECT file_version_id, paper_id, sha256, original_relative_path, created_at
FROM file_versions;

CREATE TABLE document_generations_new (
    document_generation_id TEXT PRIMARY KEY,
    file_version_id TEXT NOT NULL REFERENCES file_versions_new(file_version_id) ON DELETE RESTRICT,
    pipeline_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(file_version_id, pipeline_version)
) STRICT;

INSERT INTO document_generations_new
    (document_generation_id, file_version_id, pipeline_version, created_at)
SELECT document_generation_id, file_version_id, pipeline_version, created_at
FROM document_generations;

CREATE TABLE pages_new (
    page_id TEXT PRIMARY KEY,
    document_generation_id TEXT NOT NULL REFERENCES document_generations_new(document_generation_id)
        ON DELETE RESTRICT,
    page_number INTEGER NOT NULL CHECK(page_number > 0),
    text_relative_path TEXT NOT NULL CHECK(
        text_relative_path <> ''
        AND substr(text_relative_path, 1, 1) <> '/'
        AND substr(text_relative_path, -1, 1) <> '/'
        AND instr(text_relative_path, char(92)) = 0
        AND instr(text_relative_path, ':') = 0
        AND instr(text_relative_path, '//') = 0
        AND text_relative_path <> '.'
        AND text_relative_path <> '..'
        AND text_relative_path NOT GLOB './*'
        AND text_relative_path NOT GLOB '../*'
        AND text_relative_path NOT GLOB '*/.'
        AND text_relative_path NOT GLOB '*/..'
        AND text_relative_path NOT GLOB '*/./*'
        AND text_relative_path NOT GLOB '*/../*'
    ),
    UNIQUE(document_generation_id, page_number)
) STRICT;

INSERT INTO pages_new (page_id, document_generation_id, page_number, text_relative_path)
SELECT page_id, document_generation_id, page_number, text_relative_path
FROM pages;

DROP TABLE pages;
DROP TABLE document_generations;
DROP TABLE file_versions;

ALTER TABLE file_versions_new RENAME TO file_versions;
ALTER TABLE document_generations_new RENAME TO document_generations;
ALTER TABLE pages_new RENAME TO pages;
