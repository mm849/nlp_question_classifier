# BiLSTM
from torch.nn.utils.rnn import pad_sequence

class BiLSTM3(nn.Module):
    
    def __init__(self, vocab_size, embed_dim, num_class, use_glove, fine_tune_embed):
        super(BiLSTM3, self).__init__()
        self.use_glove = use_glove
        self.fine_tune_embed = fine_tune_embed

        self.hidden_size = 64
        drp = 0.3
        # n_classes = len(le.classes_)
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        self.lstm = nn.LSTM(embed_dim, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(self.hidden_size*4 , 60)
        self.linear2 = nn.Linear(60, 30)  # new linear layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(30, num_class)  # changed output size to match new layer
        self.softmax = nn.Softmax(dim=1)  # new softmax layer
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
          self.embedding.weight.requires_grad_(False)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        conc = self.relu(self.linear1(conc))
        conc = self.relu(self.linear2(conc))  # apply second linear layer and activation
        conc = self.dropout(conc)
        out = self.out(conc)
        out = self.softmax(out)  # apply softmax layer
        return out