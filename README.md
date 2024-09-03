# rag-from-scratch
## Task
This repo implements RAG from (realistic) first principles. That is, instead of using a Vector DB like `Milvus` and a modular RAG library like `LangChain`, I use lower-level tools:
* HuggingFace libraries (Transformers, SentenceTransformers, Gradio)
* FAISS

The direction I have chosen to explore is to see whether I can embed, index, and search Wikipedia on my desktop machine with 64GB RAM and an RTX 4090 (24GB GPU RAM). So, the focus has been on scaling up embedding, indexing, and retrieval.

I've successfully gotten working a naive RAG on all of Wikipedia with low latency retrieval, building the embeddings myself.

## Progress
See:
* `embed_wikipedia_chunks.ipynb` for how I chunk and embed Wikipedia.
* `subsample_embeddings.py` for sampling 10% of the embeddings without running out of memory
* `index_wikipedia_chunks.ipynb` for how I construct a FAISS index for retrieval.
* `naive_rag_inference.ipynb` for an implementation of naive RAG.

The naive RAG notebook is a demo that uses the full index on Wikipedia to answer questions. The output is compared to a plain instruct LLM without RAG. You can see some interesting examples in the notebook. I also examined whether the retrieved chunks were relevant and discovered the importance of setting a high `K` and `nprobes`.

## Models
* `Mistral 7B-instruct v0.3` in NF4 quantization for the frozen LLM.
* `multi-qa-MiniLM-L6-cos-v1` from SentenceTransformers for the document and query encoding.

## TODO
Currently I'm working on:
* Evaluation. I think this is the most important next step!
* A few bells-and-whistles like HyDE, Step-Back Prompting, etc.
* A Gradio demo to enter queries and compare methods.

A challenge has been running out of CPU memory while manipulating the NumPy arrays for the embeddings, which results in the Python process being killed.

My workaround is to avoid joining the embeddings into a single (or a few) matrices. Instead, I keep the embeddings in a number of smaller files, sample 10% of the data for training the index on the GPU, then add them one file at a time to the index.
