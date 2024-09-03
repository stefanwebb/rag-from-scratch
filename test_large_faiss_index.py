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

index = faiss.read_index("wikipedia-en.index")
print(index.ntotal)
