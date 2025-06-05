#!/usr/bin/env python3
from scipy.sparse import coo_matrix, lil_matrix
import numpy as np
import pickle

def main():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    print(vocab_size)

    # Initialize a sparse matrix in LIL format (better for incremental construction)
    cooc = lil_matrix((vocab_size, vocab_size), dtype=np.int32)

    counter = 1
    for fn in ['train_pos.txt', 'train_neg.txt']:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        cooc[t, t2] += 1

                if counter % 10000 == 0:
                    print(counter)
                counter += 1

    # Convert to COO format before saving
    cooc = cooc.tocoo()
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open('cooc.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()

