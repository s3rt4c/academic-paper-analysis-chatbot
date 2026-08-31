CREATE TABLE pages_new (
    page_id TEXT PRIMARY KEY,
    document_generation_id TEXT NOT NULL REFERENCES document_generations(document_generation_id)
        ON DELETE RESTRICT,
    page_number INTEGER NOT NULL CHECK(page_number > 0),
    text_relative_path TEXT CHECK(
        text_relative_path IS NULL OR (
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
        )
    ),
    physical_page_index INTEGER CHECK(physical_page_index IS NULL OR physical_page_index >= 0),
    printed_page_label TEXT,
    canonical_text TEXT,
    canonical_text_sha256 TEXT CHECK(
        canonical_text_sha256 IS NULL OR length(canonical_text_sha256) = 64
    ),
    parser_profile_sha256 TEXT CHECK(
        parser_profile_sha256 IS NULL OR length(parser_profile_sha256) = 64
    ),
    page_width_points REAL CHECK(page_width_points IS NULL OR page_width_points > 0.0),
    page_height_points REAL CHECK(page_height_points IS NULL OR page_height_points > 0.0),
    source_page_rotation_degrees INTEGER,
    extraction_quality TEXT CHECK(
        extraction_quality IS NULL OR extraction_quality IN (
            'adequate_native_text', 'low_native_text', 'empty_native_text'
        )
    ),
    needs_ocr INTEGER CHECK(needs_ocr IS NULL OR needs_ocr IN (0, 1)),
    UNIQUE(document_generation_id, page_number),
    UNIQUE(document_generation_id, physical_page_index)
) STRICT;

INSERT INTO pages_new (page_id, document_generation_id, page_number, text_relative_path)
SELECT page_id, document_generation_id, page_number, text_relative_path
FROM pages;

DROP TABLE pages;
ALTER TABLE pages_new RENAME TO pages;

CREATE TABLE generation_publications (
    file_version_id TEXT PRIMARY KEY REFERENCES file_versions(file_version_id) ON DELETE RESTRICT,
    document_generation_id TEXT NOT NULL UNIQUE
        REFERENCES document_generations(document_generation_id) ON DELETE RESTRICT
) STRICT;

CREATE TRIGGER generation_publications_file_version_matches_generation_insert
BEFORE INSERT ON generation_publications
WHEN NOT EXISTS (
    SELECT 1 FROM document_generations
    WHERE document_generation_id = NEW.document_generation_id
      AND file_version_id = NEW.file_version_id
)
BEGIN
    SELECT RAISE(ABORT, 'active generation does not belong to its file version');
END;

CREATE TRIGGER generation_publications_file_version_matches_generation_update
BEFORE UPDATE OF file_version_id, document_generation_id ON generation_publications
WHEN NOT EXISTS (
    SELECT 1 FROM document_generations
    WHERE document_generation_id = NEW.document_generation_id
      AND file_version_id = NEW.file_version_id
)
BEGIN
    SELECT RAISE(ABORT, 'active generation does not belong to its file version');
END;

CREATE TABLE page_anchors (
    page_anchor_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL,
    page_id TEXT NOT NULL REFERENCES pages(page_id) ON DELETE RESTRICT,
    char_start INTEGER NOT NULL CHECK(char_start >= 0),
    char_end INTEGER NOT NULL CHECK(char_end > char_start),
    anchor_text TEXT NOT NULL CHECK(anchor_text <> ''),
    anchor_text_sha256 TEXT NOT NULL CHECK(length(anchor_text_sha256) = 64),
    boxes_sha256 TEXT NOT NULL CHECK(length(boxes_sha256) = 64),
    x0 REAL NOT NULL,
    top REAL NOT NULL,
    x1 REAL NOT NULL,
    bottom REAL NOT NULL,
    UNIQUE(page_id, char_start, char_end)
) STRICT;

CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    document_generation_id TEXT NOT NULL
        REFERENCES document_generations(document_generation_id) ON DELETE RESTRICT,
    page_id TEXT NOT NULL REFERENCES pages(page_id) ON DELETE RESTRICT,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    start_offset INTEGER NOT NULL CHECK(start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK(end_offset > start_offset),
    chunk_text TEXT NOT NULL CHECK(chunk_text <> ''),
    lexical_word_count INTEGER NOT NULL CHECK(lexical_word_count > 0),
    processing_profile_id TEXT NOT NULL CHECK(processing_profile_id <> ''),
    UNIQUE(page_id, ordinal),
    UNIQUE(page_id, start_offset, end_offset, processing_profile_id)
) STRICT;

CREATE TRIGGER chunks_page_matches_generation_insert
BEFORE INSERT ON chunks
WHEN NOT EXISTS (
    SELECT 1 FROM pages
    WHERE page_id = NEW.page_id
      AND document_generation_id = NEW.document_generation_id
)
BEGIN
    SELECT RAISE(ABORT, 'chunk page does not belong to its document generation');
END;

CREATE TRIGGER chunks_page_matches_generation_update
BEFORE UPDATE OF document_generation_id, page_id ON chunks
WHEN NOT EXISTS (
    SELECT 1 FROM pages
    WHERE page_id = NEW.page_id
      AND document_generation_id = NEW.document_generation_id
)
BEGIN
    SELECT RAISE(ABORT, 'chunk page does not belong to its document generation');
END;

CREATE INDEX page_anchors_page_range_index ON page_anchors(page_id, char_start, char_end);
CREATE INDEX chunks_generation_page_ordinal_index
    ON chunks(document_generation_id, page_id, ordinal);

CREATE VIRTUAL TABLE chunk_fts USING fts5(
    chunk_id UNINDEXED,
    chunk_text,
    tokenize = 'unicode61 remove_diacritics 2'
);
