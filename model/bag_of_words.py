# Bag of Words

import torch.nn as nn
import torch.nn.functional as F
class QuestionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, use_glove, fine_tune_embed):
        super().__init__()
        self.use_glove = use_glove
        self.fine_tune_embed = fine_tune_embed

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True) # Bag-of-Words
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        # random initialised embedding
        if self.use_glove == False:
          self.embedding.weight.data.uniform_(-initrange, initrange) 
        # Glove embedding
        else:
          self.embedding.weight = nn.Parameter(torch.tensor(glove_weights_matrix, dtype=torch.float32))
        
        # If freeze the embedding weight
        if self.fine_tune_embed == False: 
          self.embedding.requires_grad_(False)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()


    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = self.fc(embedded)
        return x