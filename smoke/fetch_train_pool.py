"""Extract a pool of *spatial* train-split rows from SAT_train.parquet.

Why this exists instead of `load_dataset(..., streaming=True)`:
  SAT_train.parquet has ONE row group of 172,384 rows / 4.96 GiB. Both
  `datasets` streaming and the HF datasets-server fail on it with
  `ArrowNotImplementedError: Nested data conversions not implemented for
  chunked array outputs` (the server also reports
  `TooBigRowGroupsError: first row group 5204829455 > limit 300000000`).
  `pq.ParquetFile.iter_batches` reads incrementally and does work.

Also: the train split is ORDERED. Rows 0..127,404 are all question_type
"other"; the spatial types start at row 127,405. Sampling the head would
yield zero spatial questions, so we start at SPATIAL_START.

The pool is cached to a gitignored pickle so reruns are instant.
"""
import argparse
import os
import pickle
import random
import time

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

PARQUET = "datasets/array/SAT/SAT_train.parquet"
SPATIAL_START = 127405
CACHE = "data/train_spatial_pool.pkl"
COLUMNS = ["image_bytes", "question", "answers", "question_type", "correct_answer"]


def fetch(pool_size, start=SPATIAL_START, batch_size=256, cache=CACHE):
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            pool = pickle.load(f)
        if len(pool) >= pool_size:
            print(f"cache hit: {cache} ({len(pool)} rows)")
            return pool[:pool_size]

    fs = HfFileSystem()
    t0 = time.time()
    pool, idx = [], 0
    with fs.open(PARQUET, "rb") as fh:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(batch_size=batch_size, columns=COLUMNS):
            n = batch.num_rows
            if idx + n > start:
                for r in batch.to_pylist():
                    if idx >= start:
                        pool.append(r)
                    idx += 1
                    if len(pool) >= pool_size:
                        break
            else:
                idx += n
            if len(pool) >= pool_size:
                break
            if idx % 20480 == 0:
                print(f"  ...row {idx}/{start} ({time.time()-t0:.0f}s)", flush=True)
    print(f"collected {len(pool)} rows from index {start} in {time.time()-t0:.0f}s")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(pool, f)
    return pool


def sample(pool, n, seed=0):
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=256)
    a = ap.parse_args()
    p = fetch(a.pool)
    import collections
    print("pool question_type:", dict(collections.Counter(r["question_type"] for r in p)))
    print("n_images distribution:", dict(collections.Counter(len(r["image_bytes"]) for r in p)))
