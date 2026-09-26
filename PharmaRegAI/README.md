
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
