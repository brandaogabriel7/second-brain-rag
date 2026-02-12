import pytest
from ingest.obsidian import ObsidianReader


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "tagged.md").write_text(
        "---\ntags:\n  - python\n  - rag\n---\nSome content about RAG"
    )
    (tmp_path / "plain.md").write_text("Just plain content, no frontmatter")
    (tmp_path / "subfolder").mkdir()
    (tmp_path / "subfolder" / "nested.md").write_text("Nested note content")
    (tmp_path / "with-headings.md").write_text(
        "# Main Title\nIntro paragraph\n## Section One\nFirst section\n### Subsection\nDetails here"
    )
    (tmp_path / "body-tags.md").write_text(
        "# My Note\nSome text #learning and #coding here\n## Another heading\nMore #python stuff"
    )
    return tmp_path


class TestReadAllVaultNotes:
    def test_finds_all_notes(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        assert len(notes) == 5

    def test_finds_nested_notes(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        titles = [n["title"] for n in notes]
        assert "nested" in titles

    def test_invalid_vault_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ObsidianReader(str(tmp_path / "nonexistent"))


class TestFolderFiltering:
    def test_skips_hidden_folders(self, vault):
        hidden = vault / ".obsidian"
        hidden.mkdir()
        (hidden / "config.md").write_text("should be skipped")

        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        assert not any(n["title"] == "config" for n in notes)

    def test_skips_underscore_folders(self, vault):
        templates = vault / "_templates"
        templates.mkdir()
        (templates / "template.md").write_text("should be skipped")

        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        assert not any(n["title"] == "template" for n in notes)

    def test_skips_excalidraw_folders(self, vault):
        excalidraw = vault / "Excalidraw"
        excalidraw.mkdir()
        (excalidraw / "drawing.md").write_text("should be skipped")

        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        assert not any(n["title"] == "drawing" for n in notes)


class TestFrontmatterParsing:
    def test_parses_frontmatter_tags(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        tagged = next(n for n in notes if n["title"] == "tagged")
        assert "python" in tagged["frontmatter"]["tags"]
        assert "rag" in tagged["frontmatter"]["tags"]

    def test_no_frontmatter_returns_empty_dict(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        plain = next(n for n in notes if n["title"] == "plain")
        assert plain["frontmatter"] == {}

    def test_empty_frontmatter_returns_empty_dict(self, tmp_path):
        (tmp_path / "empty-fm.md").write_text("---\n---\nBody here")
        reader = ObsidianReader(str(tmp_path))
        notes = reader.read_all_vault_notes()
        assert notes[0]["frontmatter"] == {}

    def test_malformed_single_delimiter_does_not_crash(self, tmp_path):
        (tmp_path / "bad.md").write_text("---\nno closing delimiter")
        reader = ObsidianReader(str(tmp_path))
        notes = reader.read_all_vault_notes()
        assert len(notes) == 1
        assert "no closing delimiter" in notes[0]["content"]

    def test_bad_yaml_does_not_crash(self, tmp_path):
        (tmp_path / "bad-yaml.md").write_text("---\n: :\n  - [\n---\nBody")
        reader = ObsidianReader(str(tmp_path))
        notes = reader.read_all_vault_notes()
        assert len(notes) == 1


class TestContentParsing:
    def test_content_excludes_frontmatter(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        tagged = next(n for n in notes if n["title"] == "tagged")
        assert "tags:" not in tagged["content"]
        assert "RAG" in tagged["content"]

    def test_plain_note_has_full_content(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        plain = next(n for n in notes if n["title"] == "plain")
        assert plain["content"] == "Just plain content, no frontmatter"


class TestTagExtraction:
    def test_extracts_frontmatter_tags(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        tagged = next(n for n in notes if n["title"] == "tagged")
        assert "python" in tagged["tags"]
        assert "rag" in tagged["tags"]

    def test_extracts_body_tags(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        note = next(n for n in notes if n["title"] == "body-tags")
        assert "learning" in note["tags"]
        assert "coding" in note["tags"]
        assert "python" in note["tags"]

    def test_headings_not_extracted_as_tags(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        note = next(n for n in notes if n["title"] == "with-headings")
        # Headings like "# Main Title" should not produce tags
        assert "Main" not in note["tags"]
        assert "Section" not in note["tags"]
        assert "Subsection" not in note["tags"]

    def test_no_tags_returns_empty_list(self, tmp_path):
        (tmp_path / "no-tags.md").write_text("No tags at all")
        reader = ObsidianReader(str(tmp_path))
        notes = reader.read_all_vault_notes()
        assert notes[0]["tags"] == []

    def test_body_tags_mixed_with_headings(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        note = next(n for n in notes if n["title"] == "body-tags")
        # Should extract inline tags but not heading text
        assert "learning" in note["tags"]
        assert "My" not in note["tags"]
        assert "Another" not in note["tags"]


class TestNoteShape:
    def test_note_has_all_expected_keys(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        for note in notes:
            assert "title" in note
            assert "path" in note
            assert "frontmatter" in note
            assert "tags" in note
            assert "content" in note

    def test_title_has_no_extension(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        for note in notes:
            assert not note["title"].endswith(".md")

    def test_path_is_relative(self, vault):
        reader = ObsidianReader(str(vault))
        notes = reader.read_all_vault_notes()
        nested = next(n for n in notes if n["title"] == "nested")
        assert nested["path"] == "subfolder/nested.md"
