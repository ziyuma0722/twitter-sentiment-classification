## Twitter Sentiment Classification

report: https://drive.google.com/file/d/1bM2DZDiDUhuBgVz6Ee02rZRBcYg3XTD6/view?usp=drive_link

presentation: https://docs.google.com/presentation/d/1XFqeXoJIQinKPx3d6Z7qh3sa-iCLIXWk/edit?usp=drive_link&ouid=110187690232131095658&rtpof=true&sd=true


### Introduction of folders and files:

1. Folder "cooc_matrix_code":
   - Code used to generate the co-occurence matrix with the vocabulary of the 20000 most frequent words in the preprocessed datasets.
   - Detailed instructions for generating the co-occurrence matrix can be found in the README file in the folder.

2. Folder "materials" https://drive.google.com/file/d/1ZKzAuCvCJnrQm8k3Wi4x4KFZXtrO_Qmd/view?usp=drive_link :
   - "cooc_preprocess_20000.pkl" and "vocab_preprocess_20000.pkl" used by Training_GloVe.ipynb to train the cooc matrix.
   - "english": stopword used in Preprocessing.ipynb
   - "glove.twitter.27B.200d" is the stanford pre-trained 200d twitter word vectors.
   - "glove_embeddings_20000.txt" is the gloVe embedding trained by ourselves.
   - "train_neg_full_cleaned.txt", "train_pos_full_cleaned.txt", and "test_cleaned.txt" are the preprocessed "train_neg_full.txt", "train_pos_full.txt" and "test_data.txt".

3. Code "Preprocessing.ipynb": code for preprocessing the data.
4. Code "Training_GloVe.ipynb": code for training our own GloVe embedding.
5. Code "Submission.ipynb": code for creating the submission file for models without BERT/RoBERTa.

6. Folder "Simple_linear_baselines_code": simple linear baseline models of the III.A part of our report.
    1. Logistic Regression with Bag-of-words:
       - "BoW_Logistic_Regression.ipynb"
       - "TF_IDF_Logistic_Regression.ipynb"
       - "TF_IDF_Bigram_Logistic_Regression.ipynb"
       - "TF_IDF_n_gram_Logistic_Regression.ipynb"
    2. Logistic Regression with GloVe Embeddings:
       - "GloVe_Logistic_Regression.ipynb"

7. Folder "CNN": CNN models of III.B.(1) of our report
   - 1D CNN: "1D_CNN.ipynb"
   - Multilayer_CNN:  "Multilayer_CNN.ipynb", architecture that uses stacked convolutional layers, not mentioned in the report due to limited report length, didn't achieve good result
               

8. Folder "LSTM": LSTM models of III.B.(2) of our report
   - Folder "unidirectional_LSTM": unidirectional LSTM models with 1-3 LSTM layers
   - Folder "bidirectional_LSTM": bidirectional LSTM models with 1-2 BiLSTM layers

9. Folder "LSTM_CNN": LSTM and CNN hybrid architectures of III.C.(1) of our report
    - Code "uni_LSTM_(three)CNN": unidirectional LSTM layer appended with CNN of 3 kernel sizes (with maxpooling, concatenation...)
    - Code "BiLSTM_(three)CNN": bidirectional LSTM layer appended with CNN of 3 kernel sizes (with maxpooling, concatenation...)
    - Code "two_BiLSTM_(three)CNN": two bidirectional LSTM layers appended with CNN of 3 kernel sizes (with maxpooling, concatenation...)
    - Code "BiLSTM_(one)CNN": bidirectional LSTM layer appended with a simple CNN(one kernel size)
    - Code "two_BiLSTM_(one)CNN": two bidirectional LSTM layers appended with a simple CNN(one kernel size)
    - Code "CNN_BiLSTM": a simple CNN(one kernel size) appended with a bidirectional LSTM layer

10. Folder "LSTM_Attention": LSTM and Attention hybrid architectures of III.C.(2) of our report<br>
       - Code "BiLSTM_SelfAttention": bidirectional LSTM appended with a SeqSelfAttention layer
       - Code "BiLSTM_MultiheadAttention": bidirectional LSTM appended with a MultiHeadAttention layer
       - Code "BiLSTM_MultiheadAttention_AddNorm": bidirectional LSTM appended with a MultiHeadAttention layer, then add and normalization
       - Code "twoBiLSTM_MultiheadAttention": two bidirectional LSTM layers appended with a MultiHeadAttention layer

11. Folder "CNN_LSTM_Attention": Models using the combinations of CNN, LSTM, and Attention, III.C.(3) of our report<br>
       - Code "ABCDM": the ABCDM model
       - Code "DNN_MHAT": the DNN_MHAT model

12. Folder "BERT_RoBERTa": Baseline models using BERT or BoBERTa as embeddings, part III.C.(4) of our report<br>
       - Code "BERT": "bert-base-uncased" appending a dropout layer
       - Code "Twitter_RoBERTa": "Twitter-RoBERTa-Base-Sentiment" appending a dropout layer
       - Code "Twitter_RoBERTa_BiLSTM": "Twitter-RoBERTa-Base-Sentiment" in combination with a Bi-LSTM layer

13. Folder "NOVEL_MODELS": novel models created by ourselves, models introduced in the "IV. THE NOVEL MODEL" part of our report<br>
       - Code "MODEL1": the "MODEL1" model introduced in the report
       - Code "MODEL2": the "MODEL2" model introduced in the report
       - Code "MODEL3" and "MODEL4": other BiLSTM-CNN-MultiHeadAttention variants that we tried, but MODEL1 and MODEL2 have better performance.
       - Code "MODEL1_RoBERTa": "Twitter-RoBERTa-Base-Sentiment" in combination with "MODEL1"
       - Code "MODEL2_RoBERTa": "Twitter-RoBERTa-Base-Sentiment" in combination with "MODEL2"

14. Code “LoRA": use LoRA to finetune the Twitter-RoBERTa model


### Running Code
We use Google Colab to run the code.<br>
- For all linear baseline models, CPU could be used to run the code.<br>
- For all neural network models, L4 GPU/T4 GPU were used to train the model.<br>
- For all neural network models including BERT/RoBERTa, L4 was used to train the model.

To test or run the code, upload the corresponding materials in the "materials" folder into google drive and the jupyter notebook file into google colab.<br>
- For BERT/RoBERTa related files, we use transformers==4.37.2 and tensorflow==2.15.0 (default for the current google colab) to ensure compatibility. Besides, for those files, submission file production code is self-contained.<br>
- For other models, first use the corresponding file to train the model (model parameters are saved every epoch). Then use the Submission.ipynb file, replace the model part with the your current model, to produce the submission file.

For some reason, the checkpoint files need to be renamed in google colab to make everything work properly. For each epoch, an index and a data file will be saved. As an example, for epoch1, we have "cp-0001.ckpt".index" and 
"cp-0001.ckpt".data-00000-of-00001" files. We need to remove the " in the names, so they become "cp-0001.ckpt.index" and "cp-0001.ckpt.data-00000-of-00001". 

Then the code<br>
#checkpoint_path = '/content/drive/MyDrive/LSTM_CNN/cp-0001.ckpt'<br>
#model.load_weights(checkpoint_path)<br>
will automatically find the .index and .data-00000-of-00001 files for LSTM_CNN's cp-0001.ckpt and load the weights.
