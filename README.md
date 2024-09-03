# rag-from-scratch
## Task
This repo implements RAG from (realistic) first principles. That is, instead of using a Vector DB like `Milvus` and a modular RAG library like `LangChain`, I use lower-level tools:
* HuggingFace libraries (Transformers, SentenceTransformers, Gradio)
* FAISS

The direction I have chosen to explore is to see whether I can embed, index, and search Wikipedia on my desktop machine with 64GB RAM and an RTX 4090 (24GB GPU RAM). So, the focus has been on scaling up embedding, indexing, and retrieval.

## Progress
See:
* `embed_wikipedia_chunks.ipynb` for how I chunk and embed Wikipedia.
* `index_wikipedia_chunks.ipynb` for how I construct a FAISS index for retrieval.
* `naive_rag_inference.ipynb` for an implementation of naive RAG.
The naive RAG notebook is a demo that uses indexed chunks on the Wikipedia article for "Abraham Lincoln." After finishing the indexing notebook, I will update the RAG notebook to use all of Wikipedia.

## Models
* `Mistral 7B-instruct v0.3` in NF4 quantization for the frozen LLM.
* `multi-qa-MiniLM-L6-cos-v1` from SentenceTransformers for the document and query encoding.

## TODO
Currently I'm working on:
* Building scalable FAISS index.
* Scalably searching the document chunks.
* A few bells-and-whistles like Step-Back Prompting.