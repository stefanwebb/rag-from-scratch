import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
import pickle
import gc

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

print('Sampling 10pc of Embeddings')
gc.collect()
embeddings = []
ratio = 0.1
for idx, file in enumerate(files):
    print(f"File {idx} of 41")
    embeddings_file = os.path.join(embeddings_path, f'embeddings-{idx:05d}-of-00041.npy')
    X = np.load(embeddings_file)
    np.random.shuffle(X)

    # NOTE: It's important to wrap in np.array so we have a deep copy and can free memory
    Xsub = np.array(X[:int(ratio * X.shape[0])])
    embeddings.append(Xsub)
    del X
    gc.collect()

print('Concatenating Subset')
X = np.concatenate(embeddings, axis=0)
gc.collect()

print('Saving Subset')
embeddings_file = os.path.join(embeddings_path, f'subset-embeddings.npy')
with open(embeddings_file, 'wb') as f:
    np.save(f, X)

del X
gc.collect()