"""
    My Jupyter notebook kernel crashes when loading large NumPy arrays.

    Seeing if same thing happens in regular Python script.
"""
import gc
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
import pickle

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
import transformers
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

device = torch.device('cuda')

documents_path = '/home/stefanwebb/data/wikimedia/wikipedia/20231101.en'
embeddings_path = '/home/stefanwebb/embeddings/wikimedia/wikipedia/20231101.en'
files = [f"train-{idx:05d}-of-00041.parquet" for idx in range(41)]
batch_size = 1024

# embeddings = []
# for idx, file in enumerate(files[(len(files)//2):]):
#     file_id = idx + len(files)//2
#     print(f"File {file_id} of 41")
#     embeddings_file = os.path.join(embeddings_path, f'embeddings-{file_id:05d}-of-00041.npy')
#     embeddings.append(np.load(embeddings_file))

# gc.collect()

# embeddings_file = os.path.join(embeddings_path, f'embeddings-matrix-2-of-2.npy')
# with open(embeddings_file, 'wb') as f:
#     np.save(f, np.concatenate(embeddings, axis=0))

embed_dim = 384
index = faiss.index_factory(embed_dim, "PCA64,IVF16384_HNSW32,Flat")
index_ivf = faiss.extract_index_ivf(index)
clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
index_ivf.clustering_index = clustering_index

file1 = os.path.join(embeddings_path, f'embeddings-matrix-1-of-2.npy')
file2 = os.path.join(embeddings_path, f'embeddings-matrix-2-of-2.npy')

print("Loading 1/2 of embeddings matrix")
E1 = np.load(file1)
print(E1.shape)
# E2 = np.load(file2)
# gc.collect()
# E = np.concatenate([E1, E2], axis=0)
gc.collect()

print("Training index")
# NOTE: We don't have enough GPU memory to train on all embeddings
# and CPU training is too slow
index.train(E1[0:1000000])
gc.collect()

print("Adding to index")
for idx in range(24):
    print(idx,'of 24')
    lo = min(E1.shape[0], int(idx * 10**6))
    hi = min(E1.shape[0], int((idx + 1) * 10**6))
    index.add(E1[lo:hi])

del E1
gc.collect()

print("Saving index")
faiss.write_index(index, "wikipedia-en.index")