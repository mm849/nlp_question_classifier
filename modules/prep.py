# Read train dataset
class ReadDataset(torch.utils.data.Dataset):
    def __init__(self, filename, target ,transform = None):
        with open(filename, 'r', encoding='iso-8859-1') as f:
            self.sentences = f.readlines()
            self.labels = []
            for sentence in self.sentences:
                label, text = sentence.strip().split(" ", 1)
                coarse = label.split(':')[0]
                if target == 'coarse':
                  self.labels.append(coarse)
                elif target == 'fine':
                  self.labels.append(label)
        self.transform = transform
    
    def __getitem__(self, index):
        # Get the sentence and label at the given index
        sentence = self.sentences[index].strip().split(' ', 1)[1]
        sentence = sentence.strip().lower().split()
              
        return sentence, self.labels[index]
        
    def __len__(self):
        return len(self.sentences)
    

# Pipelines
def text_pipeline(sentence, train_vocab):
  indexed_sentence = []
  for w in sentence:
    if w in train_vocab:
      indexed_sentence.append(train_vocab[w])
    else:
      indexed_sentence.append(train_vocab['<unk>'])
  return indexed_sentence

def coarse_label_pipeline(label):
  if label == "ABBR":
    return 0
  elif label == "DESC":
    return 1
  elif label == "ENTY":
    return 2
  elif label == "HUM":
    return 3
  elif label == "LOC":
    return 4
  elif label == "NUM":
    return 5

def fine_label_pipeline(label):
  if label == "ABBR:abb":
    return 0
  elif label == "ABBR:exp":
    return 1
  elif label == "ENTY:animal":
    return 2
  elif label == "ENTY:body":
    return 3
  elif label == "ENTY:color":
    return 4
  elif label == "ENTY:cremat":
    return 5
  elif label == "ENTY:currency":
    return 6
  elif label == "ENTY:dismed":
    return 7
  elif label == "ENTY:event":
    return 8
  elif label == "ENTY:food":
    return 9
  elif label == "ENTY:instru":
    return 10
  elif label == "ENTY:lang":
    return 11
  elif label == "ENTY:letter":
    return 12
  elif label == "ENTY:other":
    return 13
  elif label == "ENTY:plant":
    return 14
  elif label == "ENTY:product":
    return 15
  elif label == "ENTY:religion":
    return 16
  elif label == "ENTY:sport":
    return 17
  elif label == "ENTY:substance":
    return 18
  elif label == "ENTY:symbol":
    return 19
  elif label == "ENTY:techmeth":
    return 20
  elif label == "ENTY:termeq":
    return 21
  elif label == "ENTY:veh":
    return 22
  elif label == "ENTY:word":
    return 23
  elif label == "DESC:def":
    return 24
  elif label == "DESC:desc":
    return 25
  elif label == "DESC:manner":
    return 26
  elif label == "DESC:reason":
    return 27
  elif label == "HUM:gr":
    return 28
  elif label == "HUM:ind":
    return 29
  elif label == "HUM:title":
    return 30
  elif label == "HUM:desc":
    return 31
  elif label == "LOC:city":
    return 32
  elif label == "LOC:country":
    return 33
  elif label == "LOC:mount":
    return 34
  elif label == "LOC:other":
    return 35
  elif label == "LOC:state":
    return 36
  elif label == "NUM:code":
    return 37
  elif label == "NUM:count":
    return 38
  elif label == "NUM:date":
    return 39
  elif label == "NUM:dist":
    return 40
  elif label == "NUM:money":
    return 41
  elif label == "NUM:ord":
    return 42
  elif label == "NUM:other":
    return 43
  elif label == "NUM:period":
    return 44
  elif label == "NUM:perc":
    return 45
  elif label == "NUM:speed":
    return 46
  elif label == "NUM:temp":
    return 47
  elif label == "NUM:volsize":
    return 48
  elif label == "NUM:weight":
    return 49 