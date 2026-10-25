from pathlib import Path
from uuid import uuid4

from edukg.agents.knowledge_manager.bm25_index import BM25Index
from edukg.agents.knowledge_manager.content_indexer import ContentIndexService
from edukg.core.models import Chunk, Document, DocumentType, Entity, EntityType, Provenance
from edukg.core.models.enums import ExtractionMethod
from edukg.infrastructure.faiss.faiss_vector_index import FAISSVectorIndexRepository
from edukg.infrastructure.filesystem.document_repository import FilesystemDocumentRepository
from edukg.infrastructure.filesystem.entity_repository import FilesystemEntityRepository

_DIMENSION = 4


class _FakeEmbeddingProvider:
    def embed(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def dimension(self):
        return _DIMENSION


def test_build_indexes_all_chunks_and_derives_chunk_concept_map(tmp_path: Path) -> None:
    document_repo = FilesystemDocumentRepository(base_dir=tmp_path / "documents")
    entity_repo = FilesystemEntityRepository(base_dir=tmp_path / "entities")

    document = Document(file_path=tmp_path / "doc.docx", file_hash="a" * 64, document_type=DocumentType.DOCX)
    document_repo.save(document)
    chunk = Chunk(
        document_id=document.id, sequence_index=0, text="Gradient descent minimizes loss.", char_start=0, char_end=33,
        extraction_method=ExtractionMethod.NATIVE_TEXT, token_count=8,
    )
    document_repo.save_chunks([chunk])

    entity = Entity(
        canonical_name="Gradient descent", entity_type=EntityType.CONCEPT,
        provenance=[Provenance(source_document_id=document.id, chunk_id=chunk.id, extractor_name="test", extractor_version="1.0.0")],
    )
    entity_repo.save(entity)

    service = ContentIndexService(
        document_repo, entity_repo, _FakeEmbeddingProvider(),
        dense_index=FAISSVectorIndexRepository(dimension=_DIMENSION), lexical_index=BM25Index(),
    )
    index = service.build()

    assert chunk.id in index.chunks_by_id
    assert index.chunk_concepts[chunk.id] == [entity.id]
    assert index.dense_index.search([1.0, 0.0, 0.0, 0.0], top_k=1)[0][0] == chunk.id
    assert index.lexical_index.search("gradient descent", top_k=1)[0][0] == chunk.id


def test_build_with_no_documents_returns_empty_index(tmp_path: Path) -> None:
    document_repo = FilesystemDocumentRepository(base_dir=tmp_path / "documents")
    entity_repo = FilesystemEntityRepository(base_dir=tmp_path / "entities")

    service = ContentIndexService(
        document_repo, entity_repo, _FakeEmbeddingProvider(),
        dense_index=FAISSVectorIndexRepository(dimension=_DIMENSION), lexical_index=BM25Index(),
    )
    index = service.build()

    assert index.chunks_by_id == {}
