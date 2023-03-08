from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split

import matplotlib.pyplot as plt

class TrainValTestModel():
  def __init__(self, model, collate_fn, n_epochs, batch_size, min_valid_loss) -> None:
     self.model = model
     self.collate_fn = collate_fn
     self.n_epochs = n_epochs
     self.batch_size = batch_size
     self.min_valid_loss = min_valid_loss

     # CrossEntropyLoss already contains Softmax function inside
     self.criterion = torch.nn.CrossEntropyLoss().to(device)
     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=4.0)
     self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

  def split_into_train_valid(self, train_dataset, train_ratio = 0.90):
    train_len = int(len(train_dataset) * train_ratio)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len],generator=torch.Generator().manual_seed(42))
    return sub_train_, sub_valid_

  def train(self, sub_train):
    # Set the model to train mode
    self.model.train()
    # Train model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train, batch_size=self.batch_size, shuffle=True,
                      collate_fn=self.collate_fn)
    for i, (text, offsets, cls) in enumerate(data):
        self.optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        # print(f'text: {text}')
        # print(f'offsets: {offsets}')
        # print(f'cls: {cls}')
        output = self.model(text, offsets)
        # print(f'output: {output}')
        loss = self.criterion(output, cls)
        # print(f'loss: {loss}')
        train_loss += loss.item()
        loss.backward()
        self.optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()
    # Update learning rate
    self.scheduler.step()

    return train_loss / len(sub_train), train_acc / len(sub_train)

  def validate(self, sub_valid):
    self.model.eval()
    loss = 0
    acc = 0
    valid_preds = []
    valid_labels = []
    data = DataLoader(sub_valid, batch_size=self.batch_size, collate_fn=self.collate_fn)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = self.model(text, offsets)
            loss += self.criterion(output, cls).item()
            acc += (output.argmax(1) == cls).sum().item()
            valid_preds += output.argmax(1).tolist()
            valid_labels += cls.tolist()

    return loss / len(sub_valid), acc / len(sub_valid), valid_preds, valid_labels

  def f1_score_macro(self, y_true, y_pred):
    num_labels = len(np.unique(y_true))
    f1_scores = []
    for label in range(num_labels):
        tp = np.sum(np.logical_and(y_pred == label, y_true == label))
        fp = np.sum(np.logical_and(y_pred == label, y_true != label))
        fn = np.sum(np.logical_and(y_pred != label, y_true == label))
        eps = 1e-7  # epsilon to avoid division by zero
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_scores.append(f1)
    macro_f1_score = np.mean(f1_scores)
    return macro_f1_score
  

  def train_and_evaluate(self, train_dataset, train_ratio=0.90):
    sub_train, sub_valid = self.split_into_train_valid(train_dataset, train_ratio=train_ratio)
    print('[Train & Validation]')

    train_loss_list = []
    valid_loss_list = []
    valid_preds_list = []
    valid_labels_list = []
    for epoch in range(self.n_epochs):
        start_time = time.time()
        train_loss, train_acc = self.train(sub_train)
        valid_loss, valid_acc, valid_preds, valid_labels = self.validate(sub_valid)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        self.train_loss_list = train_loss_list
        self.valid_loss_list = valid_loss_list
        valid_preds_list.append(valid_preds)
        valid_labels_list.append(valid_labels)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    # Calculate macro F1 score
    valid_preds_all = np.concatenate(valid_preds_list)
    valid_labels_all = np.concatenate(valid_labels_list)
    macro_f1_score = self.f1_score_macro(valid_labels_all, valid_preds_all)
    print('Macro F1 score: %.4f' % macro_f1_score)
  
  def loss_plot(self):
    fig = plt.figure(figsize=(6, 6))
    plt.title("Train/Validation Loss")
    plt.plot(list(np.arange(self.n_epochs)+1), self.train_loss_list, label='train')
    plt.plot(list(np.arange(self.n_epochs)+1), self.valid_loss_list, label='valid')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')

  def test(self, test_dataset):
    test_loss, test_acc, test_preds, test_labels = self.validate(test_dataset)
    print('[Test]')
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
    
    
# for BiLSTM ver2
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split

import matplotlib.pyplot as plt

class TrainValTestBiLSTMModel():
  def __init__(self, model, collate_fn, n_epochs, batch_size, min_valid_loss) -> None:
     self.model = model
     self.collate_fn = collate_fn
     self.n_epochs = n_epochs
     self.batch_size = batch_size
     self.min_valid_loss = min_valid_loss

     # CrossEntropyLoss already contains Softmax function inside
     self.criterion = torch.nn.CrossEntropyLoss().to(device)
     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
     self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

  def split_into_train_valid(self, train_dataset, train_ratio = 0.90):
    train_len = int(len(train_dataset) * train_ratio)
    sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset) - train_len])
    return sub_train_, sub_valid_

  def train(self, sub_train):
    print("train func starts")
    # Set the model to train mode
    self.model.train()
    # Train model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train, batch_size=self.batch_size, shuffle=True,
                      collate_fn=self.collate_fn)
    print("loop starts")
    for i, (text, offsets, cls) in enumerate(data):
        self.optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        #print(f'text: {text}')
        #print(f'offsets: {offsets}')
        #print(f'cls: {cls}')
        output = self.model(text)
        #print(f'output: {output}')
        loss = self.criterion(output, cls)
        #print(f'loss: {loss}')
        train_loss += loss.item()
        loss.backward()
        self.optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()
    # Update learning rate
    self.scheduler.step()

    return train_loss / len(sub_train), train_acc / len(sub_train)


  def validate(self, sub_valid):
    self.model.eval()
    loss = 0
    acc = 0
    valid_preds2 = []
    valid_labels2 = []
    data = DataLoader(sub_valid, batch_size=self.batch_size, collate_fn=self.collate_fn)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = self.model(text)
            loss = self.criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()
            valid_preds2 += output.argmax(1).tolist()
            valid_labels2 += cls.tolist()

    return loss / len(sub_valid), acc / len(sub_valid), valid_preds2, valid_labels2

  def f1_score_macro(self, y_true, y_pred):
    num_labels = len(np.unique(y_true))
    f1_scores = []
    for label in range(num_labels):
        tp = np.sum(np.logical_and(y_pred == label, y_true == label))
        fp = np.sum(np.logical_and(y_pred == label, y_true != label))
        fn = np.sum(np.logical_and(y_pred != label, y_true == label))
        eps = 1e-7  # epsilon to avoid division by zero
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_scores.append(f1)
    macro_f1_score = np.mean(f1_scores)
    return macro_f1_score    

  def train_and_evaluate(self, train_dataset, train_ratio=0.90):
    sub_train, sub_valid = self.split_into_train_valid(train_dataset, train_ratio=train_ratio)
    train_loss_list = []
    valid_loss_list = []
    valid_preds_list = []
    valid_labels_list = []
    for epoch in range(self.n_epochs):
        start_time = time.time()
        train_loss, train_acc = self.train(sub_train)
        valid_loss, valid_acc, valid_preds2, valid_labels2 = self.validate(sub_valid)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        self.train_loss_list = train_loss_list
        self.valid_loss_list = valid_loss_list
        valid_preds_list.append(valid_preds2)
        valid_labels_list.append(valid_labels2)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    # Calculate macro F1 score
    valid_preds_all = np.concatenate(valid_preds_list)
    valid_labels_all = np.concatenate(valid_labels_list)
    macro_f1_score = self.f1_score_macro(valid_labels_all, valid_preds_all)
    print('Macro F1 score: %.4f' % macro_f1_score)
  
  def loss_plot(self):
    fig = plt.figure(figsize=(6, 6))
    plt.title("Train/Validation Loss")
    plt.plot(list(np.arange(self.n_epochs)+1), self.train_loss_list, label='train')
    plt.plot(list(np.arange(self.n_epochs)+1), self.valid_loss_list, label='valid')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')

  def test(self, test_dataset):
    test_loss, test_acc, test_preds, test_labels = self.validate(test_dataset)
    print('[Test]')
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')