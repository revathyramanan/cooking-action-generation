import torch
import torch.nn.functional as F
import torch.nn as nn
from random import sample,shuffle
from tqdm import tqdm

class CookGen(nn.Module):

  def __init__(self,
               n_actions = 1008,
               h_size = 300,
               n_layers = 1,
               emb_size = 300,
               pretrained = 'bert-base-uncased'):

    super().__init__()

    self.n_layers = n_layers
    self.embeddings = nn.Embedding(1008,emb_size)
    self.input_layer = nn.Linear(emb_size,h_size)
    self.hidden_layers = nn.ModuleList([nn.Linear(h_size,h_size)
                                       for h in range(n_layers)])

    self.output_layer = nn.Linear(h_size,1008)

  def forward(self,
              input_encodings):

    n_tokens = len(input_encodings)
    input_encodings = torch.tensor(input_encodings)
    input_embeddings = self.embeddings(input_encodings)
    nn_rep = input_embeddings


    #apply the neural network
    nn_rep = F.leaky_relu(self.input_layer(nn_rep))
    for layer in self.hidden_layers:
      nn_rep = F.leaky_relu(layer(nn_rep))
    nn_rep = self.output_layer(nn_rep)

    return nn_rep[-1] #logits