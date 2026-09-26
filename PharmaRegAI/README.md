
# PharmaRegAI
### Retrieval-Augmented Regulatory Intelligence System for Pharmaceutical Supply Chains

PharmaRegAI is an AI-powered regulatory information assistant designed to retrieve relevant information from pharmaceutical regulatory documents and generate concise, source-grounded answers.

The system uses Retrieval-Augmented Generation (RAG) to reduce unsupported responses by retrieving relevant regulatory passages before generating an answer.

---

## Project Overview

Pharmaceutical supply chains operate under extensive regulatory requirements involving:

- Medicine storage
- Pharmaceutical transportation
- Good Distribution Practices
- Pharmacovigilance
- Drug recalls
- Hospital pharmacy standards
- Temperature-sensitive products
- Pharmaceutical records
- Product quality

Finding relevant information across large regulatory documents can be time-consuming.

PharmaRegAI addresses this problem by combining semantic search with a lightweight language model.

The system retrieves relevant sections from regulatory documents and provides an answer together with the source document and page number.

---

## System Architecture

```text
Regulatory PDF Documents
          |
          v
     PDF Extraction
          |
          v
      Text Chunking
          |
          v
 Sentence Transformer
    Embeddings
          |
          v
      FAISS Vector
        Database
          |
          v
   Semantic Retrieval
          |
          v
       FLAN-T5
     Answer Generation
          |
          v
 Answer + Source Pages
          |
          v
       Streamlit
       Web Interface
## Evaluation Results

The system was tested on a 20-question evaluation set spanning all 5 source documents (pharmacovigilance, distribution practices, hospital pharmacy standards, drug recalls, and storage/transport requirements).

| Metric | Score |
|---|---|
| Factual correctness | 20/20 (100%) |
| Source relevance (top-1 retrieval) | 20/20 (100%) |
| Answer completeness | 2/20 (10%) |

**Key finding:** Retrieval consistently surfaces the correct source passage, and generated answers are factually accurate, but often terse or partial rather than fully explanatory. This is a known limitation of FLAN-T5's extractive generation style on longer regulatory text. Retrieval quality is not the bottleneck here — generation is.

**Future work:** Swapping the generation stage for a larger instruction-tuned model (e.g. `flan-t5-large`, or a hosted LLM API) would likely improve completeness without changing the retrieval pipeline, since the underlying retrieved context is already accurate.
