CREATE TABLE projects (
    project_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE file_versions (
    file_version_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE RESTRICT,
    sha256 TEXT NOT NULL UNIQUE CHECK(length(sha256) = 64),
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
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE document_generations (
    document_generation_id TEXT PRIMARY KEY,
    file_version_id TEXT NOT NULL REFERENCES file_versions(file_version_id) ON DELETE RESTRICT,
    pipeline_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(file_version_id, pipeline_version)
) STRICT;

CREATE TABLE pages (
    page_id TEXT PRIMARY KEY,
    document_generation_id TEXT NOT NULL REFERENCES document_generations(document_generation_id)
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
