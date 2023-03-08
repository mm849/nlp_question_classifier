import torch

import modules.prep as prep

# Collate without padding for BoW: Coarse dataloader
from torch.utils.data import DataLoader
def collate_batch_coarse(batch):
  label_list, text_list, offsets = [], [], [0]
  for (_text, _label) in batch:
      label_list.append(prep.coarse_label_pipeline(_label))
      processed_text = torch.tensor(prep.text_pipeline(_text), dtype=torch.int64)
      text_list.append(processed_text)
      offsets.append(processed_text.size(0))
  label_list = torch.tensor(label_list, dtype=torch.int64)
  offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
  text_list = torch.cat(text_list)
  return text_list, offsets, label_list


# Collate without padding for BoW: Fine dataloader
from torch.utils.data import DataLoader
def collate_batch_fine(batch):
  label_list, text_list, offsets = [], [], [0]
  for (_text, _label) in batch:
      label_list.append(prep.fine_label_pipeline(_label))
      processed_text = torch.tensor(prep.text_pipeline(_text), dtype=torch.int64)
      text_list.append(processed_text)
      offsets.append(processed_text.size(0))
  label_list = torch.tensor(label_list, dtype=torch.int64)
  offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
  text_list = torch.cat(text_list)
  return text_list, offsets, label_list


def collate_batch_padding2_coarse(batch):
  # Let's assume that each element in "batch" is a tuple (data, label).
  # Sort the batch in the descending order
  sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

  label_list, text_list, offsets = [], [], [0]
  for (_text, _label) in sorted_batch:
      label_list.append(prep.coarse_label_pipeline(_label))
      processed_text = torch.tensor(prep.text_pipeline(_text), dtype=torch.int64)
      text_list.append(processed_text)
      offsets.append(processed_text.size(0))
  label_list = torch.tensor(label_list, dtype=torch.int64)
  offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

  # Get each sequence and pad it
  sequences = [x for x in text_list]
  sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
  # Also need to store the length of each sequence
  # This is later needed in order to unpad the sequences
  lengths = torch.LongTensor([len(x) for x in sequences])
  
  return sequences_padded, lengths, label_list




def collate_batch_padding2_fine(batch):
  # Let's assume that each element in "batch" is a tuple (data, label).
  # Sort the batch in the descending order
  sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

  label_list, text_list, offsets = [], [], [0]
  for (_text, _label) in sorted_batch:
      label_list.append(prep.fine_label_pipeline(_label))
      processed_text = torch.tensor(prep.text_pipeline(_text), dtype=torch.int64)
      text_list.append(processed_text)
      offsets.append(processed_text.size(0))
  label_list = torch.tensor(label_list, dtype=torch.int64)
  offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

  # Get each sequence and pad it
  sequences = [x for x in text_list]
  sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
  # Also need to store the length of each sequence
  # This is later needed in order to unpad the sequences
  lengths = torch.LongTensor([len(x) for x in sequences])
  
  return sequences_padded, lengths, label_list