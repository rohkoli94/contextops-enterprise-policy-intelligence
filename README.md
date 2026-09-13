# ContextOps — Enterprise Policy Intelligence Platform

ContextOps is an enterprise policy intelligence platform designed to provide
reliable, grounded answers over large and complex enterprise policy documents.

The system will evolve into a multimodal, evaluation-driven RAG platform
supporting documents containing text, tables, images and diagrams.

## Day 1

Day 1 establishes the project foundation:

- Python environment managed with uv
- FastAPI backend
- Streamlit frontend
- Versioned API structure
- Basic project structure
- Local development environment

## Current Architecture

```text
Streamlit UI : 8501
        │
        ▼
FastAPI : 8000
        │
        ▼
GET /api/v1/health
```

## Day 2 — Configuration & Environment Management

Day 2 introduces centralized application configuration and environment
management.

### What was added

- Pydantic Settings for typed application configuration
- Environment variable support
- Local `.env` support for development
- `.env.example` for documenting required configuration
- FastAPI application metadata loaded from configuration
- `.env` excluded from Git

### Configuration Flow

```text
.env / Environment Variables
            │
            ▼
    Pydantic Settings
            │
            ▼
       Application
            │
            ▼
         FastAPI
```

## Day 3 — Versioned Query API & Logging

Added the first versioned query API for ContextOps.

### Added

- `POST /api/v1/query`
- Feature-based API structure
- Separate request and response schemas
- Pydantic request validation
- Swagger/OpenAPI documentation
- Centralized logging
- Logging abstraction with `get_logger()`
- Sensitive query content is not logged

### API Flow

```text
Streamlit UI
  ↓
FastAPI
  ↓
POST /api/v1/query
  ↓
Request Validation
  ↓
query_policy()
  ↓
QueryResponse
```

### Current Endpoints

- `GET /api/v1/health`
- `POST /api/v1/query`


## Day 4 — Streamlit UI Integration

Added a Streamlit interface and connected it to the ContextOps FastAPI backend.

### Added

- Streamlit query interface
- User input validation
- FastAPI integration using HTTP
- API error handling
- Configurable backend API URL

### Flow

```text
Streamlit UI
    ↓
FastAPI Query API
    ↓
QueryResponse
    ↓
Display Answer
```

## Day 5 — LLM Provider Integration

Implemented a provider-based LLM integration architecture for ContextOps.

### Architecture

Streamlit → FastAPI → QueryService → LLMProvider → MicrosoftFoundryProvider → Microsoft Foundry

### Changes

- Added `LLMProvider` abstract contract
- Added provider-neutral `LLMRequest` and `LLMResponse`
- Added `MicrosoftFoundryProvider`
- Added `QueryService`
- Connected the query API to the LLM service
- Added Microsoft Foundry configuration
- Replaced the hardcoded query response with an LLM integration flow

### Provider Abstraction

The application service depends on the `LLMProvider` contract rather than a specific LLM provider, allowing alternative providers to be introduced without changing `QueryService`.

```text
providers/
└── llm/
    ├── base.py
    └── microsoft_foundry.py

services/
└── query_service.py
```

## Day 6 — Multimodal Document Domain & Versioning Foundation

Implemented the core domain foundation for multimodal document processing in ContextOps.

### Changes

- Added `Document` domain model
- Added `DocumentElement` domain model
- Added `ContentType` support for:
  - Text
  - Tables
  - Images
  - Charts
  - Diagrams
- Added document versioning fields
- Added page-level element tracking
- Added document and element content hashes
- Added reusable SHA-256 content hashing for both text and binary content

### Architecture

```text
Document
   ↓
DocumentElement
   ├── Text
   ├── Table
   ├── Image
   ├── Chart
   └── Diagram
```

## Day 7 — Document Storage Foundation

Implemented the storage foundation for enterprise document ingestion.

### Changes

- Added `StorageProvider` abstraction
- Added `AzureBlobStorageProvider`
- Added Azure Blob Storage configuration
- Added `azure-storage-blob` dependency
- Configured streaming-based file upload support using `BinaryIO`
- Designed the storage layer to support large file uploads without loading the entire file into application memory
- Kept storage implementation decoupled from business logic through a provider abstraction

### Architecture

```text
StorageProvider
       △
       │
AzureBlobStorageProvider
       ↓
BlobServiceClient
       ↓
ContainerClient
       ↓
BlobClient
       ↓
Azure Blob Storage
```

## Day 8 — Document Upload API & Persistence

Implemented the first end-to-end document upload workflow.

### Changes
- Added POST /api/v1/documents
- Added document upload with file, document name, categories and tags
- Added DocumentService
- Added Document and DocumentVersion persistence
- Added category and tag relationships
- Added chunk-based file hashing and size calculation
- Added Azure Blob Storage upload
- Added database transaction handling and Blob cleanup on failure
- Connected Streamlit document upload UI to FastAPI
- Added support for uploads up to 1 GB

### Architecture

```text
Streamlit UI
    ↓
POST /api/v1/documents
    ↓
Document Router
    ↓
DocumentService
    ├── PostgreSQL
    └── Azure Blob Storage
    ↓
DocumentUploadResponse
```

## Day 9 — Document Management & Version Updates

### Changes

- Added active document listing with categories and tags
- Added document version upload
- Added version update option in Streamlit
- Active document list refreshes after upload/update

### Flow

```text
Streamlit
   ↓
Documents Router
   ↓
DocumentService
   ↓
Database
```

## APIs
- GET /api/v1/documents
- POST /api/v1/documents/{document_id}/versions


## Day 10 — RAG Document Extraction Foundation
- Added DocumentExtractor abstraction
- Added DoclingDocumentExtractor
- Added Docling for structured document parsing
- Converts extracted content into DocumentElement
- Preserves document ID and page number
- Uses chunked copying to avoid loading the entire document into RAM
- Temporary file suffix is dynamically derived from the file name
- Temporary files are cleaned up after processing

```
Document Stream
      │
      ▼
DoclingDocumentExtractor
      │
      ▼
Copy to Temporary File
(1 MB buffer at a time)
      │
      ▼
Docling DocumentConverter
      │
      ▼
Structured Document
      │
      ▼
Extract Text Items
      │
      ▼
DocumentElement[]

```

## Day 11 - Multimodal Document Extraction

Implemented the document ingestion and extraction pipeline.

### Current Flow

```text
POST /documents
        |
        v
DocumentService
        |
        v
Blob Storage
        |
        v
DocumentIngestionService
        |
        v
DoclingDocumentExtractor
        |
        +-- TEXT
        |
        +-- TABLE
        |
        +-- IMAGE
              |
              v
      Vision-capable LLM
              |
              v
   IMAGE / CHART / DIAGRAM
        |
        v
DocumentElement
```

## Supported Content
- Text
- Tables
- Images
- Charts
- Diagrams


## Day 12 — Hybrid Chunking

Implemented content-type-aware hybrid chunking.

### Changes

- Added `DocumentChunk` domain model
- Added structure-aware grouping
- Added content-type-specific chunking
- Added tokenizer-aware size constraints
- Added text semantic splitting
- Added table row/column-aware splitting
- Added visual description chunking
- Preserved document/version/element lineage
- Added chunk ordering and content hashing

### Flow

```text
DocumentElement[]
        ↓
Structure-aware grouping
        ↓
Content-type strategy
        ↓
Token-aware refinement
        ↓
DocumentChunk[]
```

## Day 13 — Embeddings
- Added EmbeddingProvider abstraction.
- Added Microsoft Foundry embedding provider.
- Added single and batch embedding support.
- Added safe batching by input count and token limits.
- Added response-count validation.
- Added TiktokenCounter abstraction and implementation.
- Wired the same tokenizer into chunking and embedding.
- Separated model name from Foundry deployment name.
- Connected DocumentIngestionService:

```text
DocumentElement[]
        ↓
HybridDocumentChunker
        ↓
DocumentChunk[]
        ↓
EmbeddingProvider
        ↓
MicrosoftFoundryEmbeddingProvider
        ↓
Batch Embeddings
        ↓
EmbeddedDocumentChunk[]
```

### Key design

```text
                 TiktokenCounter
                      │
             ┌────────┴────────┐
             ↓                 ↓
      HybridChunker      EmbeddingProvider
             │                 │
             └────────┬────────┘
                      ↓
          DocumentIngestionService
```

## Day 14 — Qdrant Vector Store & Ingestion Integration

Implemented the vector storage layer and completed the ingestion path from document upload to Qdrant.

### Changes
- Added VectorStore abstraction
- Added QdrantVectorStore
- Added local Qdrant support through Docker
- Added Qdrant configuration
- Added Qdrant collection initialization
- Added cosine-distance vector configuration
- Added payload indexes for retrieval metadata
- Added tenant-aware metadata
- Added custom tenant sharding
- Added shared default shard
- Added tenant-specific dedicated shard support
- Added ShardKeyWithFallback routing
- Added Qdrant point creation
- Added Qdrant upsert
- Connected DocumentIngestionService to VectorStore
- Propagated categories and tags into DocumentChunk.metadata
- Preserved document/version/source lineage in Qdrant payload


### Current Ingestion Flow

```text

Upload API
    ↓
DocumentService
    ↓
Blob Storage
    ↓
DocumentIngestionService
    ↓
Document Extraction
    ↓
DocumentElement[]
    ↓
HybridDocumentChunker
    ↓
DocumentChunk[]
    ↓
Metadata Enrichment
    ├── tenant_id
    ├── categories
    └── tags
    ↓
Batch Embedding
    ↓
EmbeddedDocumentChunk[]
    ↓
VectorStore
    ↓
Qdrant
```


### Qdrant Architecture
```text
                    Qdrant Collection
                    contextops_documents
                            │
                    Custom Sharding
                            │
             ┌──────────────┴──────────────┐
             ↓                             ↓
       default shard               tenant-specific shard
       normal tenants                 large tenants
```

Qdrant uses one collection with tenant-aware payload filtering and custom shard routing.

### Qdrant Point

Each EmbeddedDocumentChunk becomes one Qdrant point:

```text
Point
├── id
├── vector
└── payload
    ├── tenant_id
    ├── chunk_id
    ├── document_id
    ├── document_version_id
    ├── content
    ├── element_ids
    ├── content_type
    ├── hierarchy_path
    ├── page_numbers
    ├── categories
    └── tags
```

### Metadata Strategy

PostgreSQL remains the source of truth for document lifecycle and relational metadata.

Qdrant contains the retrieval-oriented copy needed for filtering and search.

```text
PostgreSQL
    ↓
Source of truth

Qdrant
    ↓
Retrieval index
```

Categories and tags are propagated to every chunk because they are useful for retrieval-time filtering.

### Sharding vs Metadata Filtering

These are separate mechanisms:

Shard routing
→ Which physical shard should Qdrant search?

Metadata filtering
→ Which points inside that search scope match?
Local Qdrant

## Day 15 — Dense Retrieval Foundation + LangChain + LangGraph

Started the retrieval layer.

### Changes
- Added RetrievalProvider abstraction
- Added RetrievedChunk result model
- Added VectorStore.search() abstraction
- Added DenseRetriever
- Added query embedding for dense retrieval
- Added tenant-aware Qdrant retrieval
- Added metadata filtering support
- Added Qdrant ANN/vector search
- Added shard routing during retrieval
- Added foundation for LangChain retriever integration
- Began separating retrieval components from workflow orchestration

### Dense Retrieval Flow

```text
User Query
    ↓
DenseRetriever
    ↓
EmbeddingProvider
    ↓
Query Vector
    ↓
VectorStore
    ↓
Qdrant
    ↓
ANN Search
    ↓
RetrievedChunk[]
```

### Dense vs BM25

Dense retrieval:

```text
Query
 ↓
Embedding
 ↓
Vector
 ↓
ANN Search
```

BM25:

```text
Query
 ↓
Lexical / keyword search
 ↓
BM25 Results
```

Only dense retrieval requires a query embedding.

Later:

```text
             User Query
                 │
        ┌────────┴────────┐
        ↓                 ↓
      Dense              BM25
        ↓                 ↓
     Qdrant          Lexical Search
        └────────┬────────┘
                 ↓
                RRF
```

### Retrieval Result

Retrieval returns:

```text
RetrievedChunk
├── chunk
├── score
└── metadata
```

The embedding vector is not returned because retrieval needs the matched chunk, metadata and relevance score rather than the stored vector itself.


### Day 16 — Hybrid Retrieval Foundation

Implemented the production hybrid retrieval foundation by extending dense retrieval with sparse BM25 retrieval.

### Changes

- Continued dense embedding through EmbeddingProvider
- Added SparseEmbedding domain model
- Added SparseEmbeddingProvider abstraction
- Added BM25SparseEmbeddingProvider
- Added configurable BM25 model and FastEmbed cache configuration
- Added FastAPI startup initialization for the sparse embedding provider
- Added production Docker BM25 model preloading during image build
- Added Docker application setup alongside the 3-node Qdrant cluster
- Updated EmbeddedDocumentChunk to contain both dense and sparse representations
- Updated DocumentIngestionService to generate both dense and BM25 embeddings
- Updated Qdrant collection to support named dense and bm25 vectors
- Updated Qdrant points to store both dense and sparse vectors
- Added search_dense() to VectorStore
- Added search_sparse() to VectorStore
- Added search_hybrid() to VectorStore
- Updated DenseRetriever for dense-only retrieval
- Added BM25Retriever for sparse retrieval
- Added HybridRetriever
- Added Qdrant native hybrid search with RRF
- Preserved tenant-aware shard routing and metadata filtering

### Embedding Architecture

```text
DocumentChunk / User Query
          │
    ┌─────┴─────┐
    ↓           ↓
  Dense        BM25
Embedding     Sparse
 Provider     Provider
    ↓           ↓
Dense Vector  Sparse Vector
```

### Retrieval Architecture

```text
                         User Query
                             │
               ┌─────────────┴─────────────┐
               ↓                           ↓
        DenseRetriever                BM25Retriever
               ↓                           ↓
       Dense query vector          BM25 sparse query
               └─────────────┬─────────────┘
                             ↓
                     HybridRetriever
                             ↓
                    Qdrant Hybrid Search
                             ↓
                            RRF
                             ↓
                    RetrievedChunk[]
```

### Qdrant Vector Configuration

contextops_documents

```text
├── dense
│   └── 1536 dimensions / Cosine
│
└── bm25
    └── Sparse
```

Each Qdrant point now contains:

```text
Point
├── id
├── dense vector
├── bm25 sparse vector
└── payload
    ├── tenant_id
    ├── chunk_id
    ├── document_id
    ├── document_version_id
    ├── content
    ├── element_ids
    ├── content_type
    ├── hierarchy_path
    ├── page_numbers
    ├── categories
    └── tags
```

### Dense Retrieval

```text
Query
   ↓
Dense EmbeddingProvider
   ↓
Dense Query Vector
   ↓
VectorStore.search_dense()
   ↓
Qdrant Dense ANN
```

### Sparse BM25 Retrieval

```text
Query
   ↓
SparseEmbeddingProvider
   ↓
BM25 Sparse Query
   ↓
VectorStore.search_sparse()
   ↓
Qdrant BM25
```

Only the dense branch uses the dense query embedding. The BM25 branch uses a separate sparse representation.

### Hybrid Retrieval

```text
Dense Query
     +
BM25 Sparse Query
     ↓
Qdrant
     ↓
Dense Candidates + BM25 Candidates
     ↓
RRF
     ↓
Combined Ranked Results
```

The RRF fusion is handled by Qdrant's native hybrid query rather than implementing score fusion separately in the application.

### Docker

The application is now designed to run with:

```text
Docker Compose
      │
      ├── ContextOps
      │     ├── FastAPI
      │     ├── FastEmbed
      │     └── BM25 model
      │
      └── Qdrant Cluster
            ├── Node 1
            ├── Node 2
            └── Node 3
```

The Qdrant/bm25 model is preloaded during Docker image build and cached inside the application image.

### Current Retrieval Flow

```text
User Query
     ↓
Query representation
     ├── Dense vector
     └── BM25 sparse vector
     ↓
Tenant shard routing
     ↓
Metadata filtering
     ↓
Qdrant Hybrid Search
     ↓
RRF
     ↓
RetrievedChunk[]
```

## Day 17 — Query Service & Async Query Pipeline

Implemented the application query layer and connected retrieval, LangChain, Qdrant and Microsoft Foundry into an asynchronous query flow.

### Changes
- Added LangChainRetrieverAdapter and async ainvoke() retrieval
- Added async dense, BM25 and hybrid retrieval
- Added async Qdrant search with native RRF
- Added typed QueryFilter and tenant-aware filtering
- Rebuilt QueryService for retrieval → context → LLM
- Added async LLM generation with Microsoft Foundry
- Added Azure and Qdrant async client cleanup
- Added FastAPI startup composition with shared QueryService in app.state
- Updated query router and Streamlit filters
- Added aiohttp for Azure async support
- Verified Docker startup and end-to-end query flow


Query Architecture
```text
User Query
    ↓
FastAPI
    ↓
QueryService
    ↓
LangChain Retriever
    ↓
HybridRetriever
    ├── Dense Embedding
    └── BM25 Sparse Embedding
            ↓
        Qdrant Hybrid Search
            ↓
           RRF
            ↓
     Retrieved Documents
            ↓
        Context Builder
            ↓
   Microsoft Foundry LLM
            ↓
        QueryResponse
```

Application Lifecycle
```text
FastAPI Startup
      ↓
Create shared providers/services
      ↓
Store QueryService in app.state
      ↓
Requests reuse shared services
      ↓
FastAPI Shutdown
      ↓
Close Azure + Qdrant async clients
```

## Day 18 — LangGraph + State + Conversation Intelligence

Implemented the LangGraph-based query orchestration workflow.

### Query Workflow

```text
Request
  ↓
Input Validation
  ↓
Security / Guardrails
  ↓
Conversation Context
  ↓
Query Contextualization
  ↓
Cache Lookup
  ├── HIT  → Cached Response → Response
  └── MISS
       ↓
   Hybrid Retrieval
       ↓
     Reranker
       ↓
 Retrieval Validation
   ├── Sufficient → ContextOps
   └── Insufficient → Recovery / Safe Abstention
       ↓
      LLM
       ↓
 Grounding Validation
       ↓
    Response
```
Implemented
- Added typed QueryState for LangGraph workflow state.
- Added input validation node.
- Added authorization and tenant-isolation guardrails.
- Added prompt-injection detection.
- Added input PII analysis.
- Added PostgreSQL-backed conversation memory.
- Added bounded recent conversation context.
- Added rolling conversation summary support.
- Added QueryRewriter abstraction.
- Added Microsoft Foundry query contextualization.
- Added safe fallback to the original query when rewriting fails.
- Added CacheProvider abstraction.
- Added deterministic, tenant-aware, version-aware cache-key generation.
- Added cache lookup and cache-hit response routing.
- Added hybrid retrieval LangGraph node.
- Reused existing dense + BM25 + Qdrant RRF retrieval.
- Added Reranker abstraction and Day 18 pass-through implementation.
- Added RetrievalValidator abstraction and baseline implementation.
- Added retrieval confidence and conditional recovery routing.
- Added ContextOps context assembly and citation generation.
- Added configurable context document limit.
- Added configurable context token budget.
- Added asynchronous LLM generation node.
- Added GroundingValidator abstraction and baseline implementation.
- Added grounding validation node.
- Added final response node.
- Added complete LangGraph state-machine composition.
- Refactored QueryService into a thin LangGraph executor.
- Added FastAPI application-state composition.
- Added dependency injection for the shared QueryService.
- Added graph execution and workflow tests.
- Added asynchronous PostgreSQL session lifecycle for conversation persistence.
- Added aiohttp for Azure asynchronous client support.

## Day 19 — Advanced Retrieval + Confidence + Recovery

Implemented advanced retrieval controls, reranking, retrieval confidence evaluation, bounded recovery, query reformulation, re-retrieval, and safe abstention.

### Retrieval Architecture

```text
User Query
    ↓
Query Contextualization
    ↓
Hybrid Retrieval
    ├── Dense Retrieval
    └── BM25 Sparse Retrieval
            ↓
        Qdrant Hybrid Search
            ↓
           RRF
            ↓
   RetrievedChunk[]
            ↓
   Candidate Pool
   (Top-N / Top-20)
            ↓
     FastEmbed Reranker
     MS MARCO Cross-Encoder
            ↓
   RerankedChunk[]
            ↓
 Retrieval Confidence
      Evaluation
```

### Changes

- Added production `Reranker` abstraction with FastEmbed MS MARCO cross-encoder.
- Added CPU reranking and Docker model preloading.
- Added configurable top-N candidate pool and reranker score propagation.
- Added `RetrievalEvaluation` with confidence classification and retrieval signals.
- Added bounded recovery with query reformulation, broader retrieval, reranking and re-validation.
- Preserved original query, tenant ID and metadata filters during recovery.
- Added LangGraph recovery routing and safe abstention.
- Added reranking, validation and recovery tests.

### Reranking Flow

```text
Hybrid Retrieval
      ↓
Top-N Candidate Pool
      ↓
MS MARCO Reranker
      ↓
RerankedChunk[]
```

### Retrieval Confidence

```text
Retrieved Documents
        ↓
Confidence Evaluation
        ↓
STRONG / SUFFICIENT / NONE
```

Reranker scores are treated as ranking signals, not probabilities.

### Retrieval Recovery
```text
Retrieval Validation
        ↓
   Insufficient
        ↓
Query Reformulation
        ↓
Broader Retrieval
        ↓
      Rerank
        ↓
   Re-validation
      /     \
 Success   Failure
    ↓         ↓
ContextOps   Safe
    ↓       Abstention
   LLM
```

### Safe Abstention

When sufficient evidence cannot be found, weak evidence is not sent to the LLM.

"I could not find sufficient policy evidence
to answer this question reliably."



### Status

- Day 1 — Project Foundation ✅
- Day 2 — Configuration & Environment Management ✅
- Day 3 — Versioned Query API & Logging ✅
- Day 4 — Streamlit UI Integration ✅
- Day 5 — LLM Provider Abstraction & Microsoft Foundry Integration ✅
- Day 6 — Multimodal Document Domain & Versioning Foundation ✅
- Day 7 — Document Storage Foundation ✅
- Day 8 — Document Upload API & Persistence ✅
- Day 9 — Document Management & Version Updates ✅
- Day 10 — RAG Document Extraction Foundation ✅
- Day 11 - Multimodal Document Extraction ✅
- Day 12 — Hybrid Chunking ✅
- Day 13 — Embeddings ✅
- Day 14 — Qdrant Vector Store & Ingestion Integration ✅
- Day 15 — Dense Retrieval Foundation + LangChain + LangGraph ✅
- Day 16 — Hybrid Retrieval Foundation ✅
- Day 17 — Query Service & Async Query Pipeline ✅
- Day 18 — LangGraph + State + Conversation Intelligence ✅
- Day 19 — Advanced Retrieval + Confidence + Recovery
- Day 20  ✓ ContextOps / caching / observability / evaluation