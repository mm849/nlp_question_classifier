# Reproducing same results
SEED = 2023

import random
import numpy as np
import torch
import modules.prep as prep

# set seed
seed_everything()

# Read Coarse dataset
coarse_train_dataset = prep.ReadDataset("data/train_5500.label", target = "coarse")
fine_train_dataset = prep.ReadDataset("data/train_5500.label", target = "fine")

# Create vocab
train_vocab = {}
for sentence, _ in coarse_train_dataset:
  for w in sentence:
    if w not in train_vocab:
      train_vocab[w] = len(train_vocab)

train_vocab['<unk>'] = len(train_vocab)

# Load test dataet
coarse_test_dataset = prep.ReadDataset("data/TREC_10.label", target = "coarse")
fine_test_dataset = prep.ReadDataset("data/TREC_10.label", target = "fine")


glove_file = '/content/drive/MyDrive/Text Mining/glove.small.txt'
glove_dict = {}

with open(glove_file, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove_dict[word] = coefs

glove_embed_size = np.stack(glove_dict.values()).shape[1]


matrix_len = len(train_vocab)

glove_weights_matrix = np.zeros((matrix_len, glove_embed_size))
words_found = 0

for i, word in enumerate(train_vocab):
  try: 
    glove_weights_matrix[i] = glove_dict[word]
    words_found += 1
  except KeyError:
    glove_weights_matrix[i] = np.random.normal(scale=0.6, size=(glove_embed_size, ))
    
    
    
    
    
# Coarse Model
VOCAB_SIZE = len(train_vocab)
EMBED_DIM = glove_embed_size # 300
NUM_CLASS_coarse = len(set(coarse_train_dataset.labels))
print(f'num class: {NUM_CLASS_coarse}')


# Fine Model
VOCAB_SIZE = len(train_vocab)
EMBED_DIM = glove_embed_size # 300
NUM_CLASS_fine = len(set(fine_train_dataset.labels))
print(f'num class: {NUM_CLASS_fine}')


# lr = 1e-4
dropout_prob = 0.5
# max_document_length = 100  # each sentence has until 100 words
# max_size = 5000 # maximum vocabulary size
num_hidden_nodes = 93
hidden_dim2 = 128
num_layers = 2  # LSTM layers
bi_directional = True 



# Model Implementation
from model.bag_of_words import QuestionClassifier
from model.train_val_test import TrainValTestModel

## 1: BoW, Random, Fine tune
model1 = QuestionClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASS_coarse, use_glove=False, fine_tune_embed=True).to(device)
model1_evaluator = TrainValTestModel(model = model1, collate_fn = collate_batch_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
model1_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
model1_evaluator.test(test_dataset=coarse_test_dataset)
model1_evaluator.loss_plot()


## 2: BoW, Glove, Freeze
model2 = QuestionClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASS_coarse, use_glove=True, fine_tune_embed=False).to(device)
model2_evaluator = TrainValTestModel(model = model2, collate_fn = collate_batch_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
model2_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
model2_evaluator.test(test_dataset=coarse_test_dataset)
model2_evaluator.loss_plot()


## 3: BoW, Glove, Finetune
model3 = QuestionClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASS_coarse, use_glove=True, fine_tune_embed=True).to(device)
model3_evaluator = TrainValTestModel(model = model3, collate_fn = collate_batch_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
model3_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
model3_evaluator.test(test_dataset=coarse_test_dataset)
model3_evaluator.loss_plot()


from model.bilstm import BiLSTM3
from model.train_val_test import TrainValTestBiLSTMModel

## 4: BiLSTM, Random, Fine tune
model4 = BiLSTM3(VOCAB_SIZE, EMBED_DIM, num_class=NUM_CLASS_coarse, use_glove=False, fine_tune_embed=True).to(device)
model4_evaluator = TrainValTestBiLSTMModel(model = model4, collate_fn = collate_batch_padding2_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
model4_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
model4_evaluator.test(test_dataset=coarse_test_dataset)
model4_evaluator.loss_plot()


## 5: BoW, Glove, Freeze
model5 = BiLSTM3(VOCAB_SIZE, EMBED_DIM, num_class=NUM_CLASS_coarse, use_glove=True, fine_tune_embed=False).to(device)
model5_evaluator = TrainValTestBiLSTMModel(model = model5, collate_fn = collate_batch_padding2_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
model5_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
model5_evaluator.test(test_dataset=coarse_test_dataset)
model5_evaluator.loss_plot()


## 6: BiLSTM, Glove, Finetune
model6 = BiLSTM3(VOCAB_SIZE, EMBED_DIM, num_class=NUM_CLASS_coarse, use_glove=True, fine_tune_embed=True).to(device)
model6_evaluator = TrainValTestBiLSTMModel(model = model6, collate_fn = collate_batch_padding2_coarse, n_epochs=10, batch_size=16, min_valid_loss=float('inf'))
model6_evaluator.train_and_evaluate(train_dataset=coarse_train_dataset)
model6_evaluator.test(test_dataset=coarse_test_dataset)
model6_evaluator.loss_plot()