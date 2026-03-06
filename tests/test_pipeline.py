from unittest.mock import MagicMock, patch

import pytest
from errors import ChunkingError, CriticalError, EmbeddingError
from ingest.error_collector import ErrorCollector
from ingest.pipeline import ingest, ingest_obsidian, ingest_readwise
from models import Chunk, ObsidianNote, ReadwiseHighlight


def make_note(title="Test Note", path="test.md", content="Some content"):
    return ObsidianNote(
        title=title,
        path=path,
        frontmatter={},
        tags=[],
        content=content,
    )


def make_chunk(text="chunk text", title="Test", source="test.md"):
    return Chunk(
        text=text,
        source=source,
        title=title,
        heading="",
        tags=[],
    )


def make_highlight(id="123", text="highlight text", title="Book"):
    return ReadwiseHighlight(
        id=id,
        text=text,
        title=title,
        author="Author",
        category="books",
        readwise_url=f"https://readwise.io/open/{id}",
        tags=[],
    )


class TestIngestObsidian:
    @patch("ingest.pipeline.Progress")
    @patch("ingest.pipeline.Chunker")
    @patch("ingest.pipeline.ObsidianReader")
    def test_returns_chunks_from_vault_notes(
        self, mock_reader_cls, mock_chunker_cls, mock_progress
    ):
        mock_reader = mock_reader_cls.return_value
        mock_reader.read_all_vault_notes.return_value = [
            make_note(title="Note 1"),
            make_note(title="Note 2"),
        ]
        mock_chunker = mock_chunker_cls.return_value
        mock_chunker.chunk_note.side_effect = [
            [make_chunk(text="chunk 1")],
            [make_chunk(text="chunk 2a"), make_chunk(text="chunk 2b")],
        ]

        result = ingest_obsidian(MagicMock(), "/fake/vault", ErrorCollector())

        assert len(result) == 3
        mock_reader_cls.assert_called_once_with("/fake/vault")

    @patch("ingest.pipeline.Chunker")
    @patch("ingest.pipeline.ObsidianReader")
    def test_empty_vault_returns_empty_list(self, mock_reader_cls, mock_chunker_cls):
        # Progress not used when vault is empty (returns early)
        mock_reader_cls.return_value.read_all_vault_notes.return_value = []

        result = ingest_obsidian(MagicMock(), "/fake/vault", ErrorCollector())

        assert result == []
        mock_chunker_cls.return_value.chunk_note.assert_not_called()

    @patch("ingest.pipeline.Progress")
    @patch("ingest.pipeline.Chunker")
    @patch("ingest.pipeline.ObsidianReader")
    def test_collects_chunking_errors_and_continues(
        self, mock_reader_cls, mock_chunker_cls, mock_progress
    ):
        mock_reader_cls.return_value.read_all_vault_notes.return_value = [
            make_note(title="Good Note"),
            make_note(title="Bad Note"),
            make_note(title="Another Good"),
        ]
        mock_chunker = mock_chunker_cls.return_value
        mock_chunker.chunk_note.side_effect = [
            [make_chunk(text="chunk 1")],
            ValueError("chunking failed"),
            [make_chunk(text="chunk 3")],
        ]

        collector = ErrorCollector()
        result = ingest_obsidian(MagicMock(), "/fake/vault", collector)

        assert len(result) == 2
        assert collector.count() == 1
        assert isinstance(collector.errors[0], ChunkingError)


class TestIngestReadwise:
    @patch("ingest.pipeline.ReadwiseClient")
    def test_returns_chunks_from_highlights(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.iter_highlight_pages.return_value = iter(
            [[make_highlight(id="1", text="first"), make_highlight(id="2", text="second")]]
        )

        result = ingest_readwise(MagicMock(), "test-token", request_delay=0)

        assert len(result) == 2
        assert result[0].text == "first"
        assert result[1].text == "second"

    @patch("ingest.pipeline.ReadwiseClient")
    def test_filters_empty_highlights(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.iter_highlight_pages.return_value = iter(
            [
                [
                    make_highlight(id="1", text="valid"),
                    make_highlight(id="2", text=""),
                    make_highlight(id="3", text="   "),
                ]
            ]
        )

        result = ingest_readwise(MagicMock(), "test-token", request_delay=0)

        assert len(result) == 1
        assert result[0].text == "valid"

    @patch("ingest.pipeline.ReadwiseClient")
    def test_handles_pagination(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.iter_highlight_pages.return_value = iter(
            [
                [make_highlight(id="1", text="page 1")],
                [make_highlight(id="2", text="page 2")],
            ]
        )

        result = ingest_readwise(MagicMock(), "test-token", request_delay=0)

        assert len(result) == 2

    @patch("ingest.pipeline.ReadwiseClient")
    def test_empty_response_returns_empty_list(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.iter_highlight_pages.return_value = iter([])

        result = ingest_readwise(MagicMock(), "test-token", request_delay=0)

        assert result == []


class TestIngest:
    @patch("ingest.pipeline.ingest_readwise")
    @patch("ingest.pipeline.ingest_obsidian")
    def test_ingests_from_both_sources_when_configured(
        self, mock_obsidian, mock_readwise
    ):
        mock_obsidian.return_value = [make_chunk(text="obs")]
        mock_readwise.return_value = [make_chunk(text="rw")]
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1, 0.2]]
        mock_store = MagicMock()

        result = ingest(
            MagicMock(),
            mock_embedder,
            mock_store,
            vault_path="/vault",
            readwise_token="token",
        )

        assert result is True
        mock_obsidian.assert_called_once()
        mock_readwise.assert_called_once()

    @patch("ingest.pipeline.ingest_readwise")
    @patch("ingest.pipeline.ingest_obsidian")
    def test_skips_source_when_not_configured(self, mock_obsidian, mock_readwise):
        mock_readwise.return_value = [make_chunk(text="rw")]
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1, 0.2]]
        mock_store = MagicMock()

        result = ingest(
            MagicMock(),
            mock_embedder,
            mock_store,
            vault_path="",
            readwise_token="token",
        )

        assert result is True
        mock_obsidian.assert_not_called()
        mock_readwise.assert_called_once()

    @patch("ingest.pipeline.ingest_readwise")
    @patch("ingest.pipeline.ingest_obsidian")
    def test_returns_false_when_no_chunks(self, mock_obsidian, mock_readwise):
        mock_obsidian.return_value = []
        mock_readwise.return_value = []
        mock_embedder = MagicMock()
        mock_store = MagicMock()

        result = ingest(
            MagicMock(),
            mock_embedder,
            mock_store,
            vault_path="/vault",
            readwise_token="token",
        )

        assert result is False
        mock_embedder.embed_batch.assert_not_called()

    @patch("ingest.pipeline.Progress")
    @patch("ingest.pipeline.ingest_readwise")
    @patch("ingest.pipeline.ingest_obsidian")
    def test_processes_chunks_in_batches(
        self, mock_obsidian, mock_readwise, mock_progress
    ):
        mock_obsidian.return_value = [make_chunk(text=f"c{i}") for i in range(5)]
        mock_readwise.return_value = []
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 3]
        mock_store = MagicMock()

        ingest(
            MagicMock(),
            mock_embedder,
            mock_store,
            vault_path="/vault",
            readwise_token="",
            batch_size=2,
        )

        # 5 chunks with batch_size=2 → 3 batches (2, 2, 1)
        assert mock_embedder.embed_batch.call_count == 3

    @patch("ingest.pipeline.Progress")
    @patch("ingest.pipeline.ingest_readwise")
    @patch("ingest.pipeline.ingest_obsidian")
    def test_recoverable_embedding_error_continues(
        self, mock_obsidian, mock_readwise, mock_progress
    ):
        mock_obsidian.return_value = [make_chunk(text=f"c{i}") for i in range(4)]
        mock_readwise.return_value = []
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.side_effect = [
            [[0.1]],  # batch 1 succeeds
            EmbeddingError(2, ValueError("API error")),  # batch 2 fails
        ]
        mock_store = MagicMock()

        result = ingest(
            MagicMock(),
            mock_embedder,
            mock_store,
            vault_path="/vault",
            readwise_token="",
            batch_size=2,
        )

        assert result is True
        assert mock_store.add_chunks.call_count == 1  # only first batch stored

    @patch("ingest.pipeline.Progress")
    @patch("ingest.pipeline.ingest_readwise")
    @patch("ingest.pipeline.ingest_obsidian")
    def test_critical_error_in_embedding_stops_processing(
        self, mock_obsidian, mock_readwise, mock_progress
    ):
        mock_obsidian.return_value = [make_chunk(text=f"c{i}") for i in range(4)]
        mock_readwise.return_value = []
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.side_effect = [
            [[0.1]],  # batch 1 succeeds
            CriticalError("Auth failed"),  # batch 2 critical failure
        ]
        mock_store = MagicMock()

        ingest(
            MagicMock(),
            mock_embedder,
            mock_store,
            vault_path="/vault",
            readwise_token="",
            batch_size=2,
        )

        # Should stop at batch 2, not try batch 3
        assert mock_embedder.embed_batch.call_count == 2

    @patch("ingest.pipeline.ingest_readwise")
    @patch("ingest.pipeline.ingest_obsidian")
    def test_critical_error_in_obsidian_continues_to_readwise(
        self, mock_obsidian, mock_readwise
    ):
        mock_obsidian.side_effect = CriticalError("Vault not found")
        mock_readwise.return_value = [make_chunk(text="rw")]
        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1, 0.2]]
        mock_store = MagicMock()

        result = ingest(
            MagicMock(),
            mock_embedder,
            mock_store,
            vault_path="/vault",
            readwise_token="token",
        )

        assert result is True
        mock_readwise.assert_called_once()
        mock_store.add_chunks.assert_called_once()
