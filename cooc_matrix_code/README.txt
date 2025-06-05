## Build the Co-occurence Matrix

train_neg.txt and train_pos.txt are the preprocessed full datasets, only renamed.

'cooc.py' requires a lot of memory since it stores all co-occurrences' coordinates and values in lists before converting them to a sparse matrix. 
We adapted this in 'cooc_save_memory.py', where all operations are performed directly on a fixed-size matrix. This adaptation saves significant memory but takes longer to execute.

To build the cooc matrix with the most frequent 20000 words, run the following commands.  (Remember to put the data files
in the correct locations)

Note that the cooc.py/cooc_save_memory.py scripts takes a few minutes to run, and displays the number of tweets processed.

- build_vocab.sh
- cut_vocab.sh
- python3 pickle_vocab.py
- python3 cooc.py/cooc_save_memory.py