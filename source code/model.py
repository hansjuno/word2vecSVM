import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

  def __init__(self, embed_num, label_num, embeddings_dim, embeddings_mode, initial_embeddings, args):
    
    # Cannot assign module before .__init__() call
    super(CNN, self).__init__()

    self.embed_num = embed_num
    self.label_num = label_num
    self.embed_dim = embeddings_dim
    self.embed_mode = embeddings_mode
    self.channel_in = 1
    self.feature_num = args.feature_num
    self.kernel_width = args.kernel_width
    self.dropout_rate = 0.5
    self.norm_limit = 3

    assert (len(self.feature_num) == len(self.kernel_width))
    self.kernel_num = len(self.kernel_width)

    # Create and initialize embeddings
    self.embeddings = nn.Embedding(self.embed_num, self.embed_dim, padding_idx=1, max_norm=self.norm_limit, norm_type=2)
    if self.embed_mode == 'non-static' or self.embed_mode == 'static' or self.embed_mode =='multichannel':
      self.embeddings.weight.data.copy_(torch.from_numpy(initial_embeddings))
      if self.embed_mode == 'static':
        self.embeddings.weight.requires_grad = False
      elif self.embed_mode == 'multichannel':
        # different to galsang's code, torchtext has already incorporated <unk> and <pad>
        # in the vocabulary of the input data
        self.embeddings2 = nn.Embedding(self.embed_num, self.embed_dim, padding_idx=1, max_norm=self.norm_limit, norm_type=2)
        self.embeddings2.weight.data.copy_(torch.from_numpy(initial_embeddings))
        self.embeddings2.weight.requires_grad = False
        self.channel_in = 2
    
    self.convs = nn.ModuleList([nn.Conv1d(self.channel_in, self.feature_num[i], 
                                          self.embed_dim*self.kernel_width[i], 
                                          stride=self.embed_dim) 
                                    for i in range(self.kernel_num)])
    
    self.linear = nn.Linear(sum(self.feature_num), self.label_num)

  def forward(self, input):
    batch_width = input.size()[1]
    x = self.embeddings(input).view(-1, 1, self.embed_dim*batch_width)
    if self.embed_mode == 'multichannel' :
      x2 = self.embeddings2(input).view(-1, 1, self.embed_dim*batch_width)
      x = torch.cat((x, x2), 1)

    conv_results = [
      F.max_pool1d(
        F.relu(self.convs[i](x)), batch_width - self.kernel_width[i] + 1 )
        .view(-1, self.feature_num[i])
      for i in range(len(self.feature_num))
    ]

    x = torch.cat(conv_results, 1)
    x = F.dropout(x, p=self.dropout_rate, training=self.training)
    x = self.linear(x)

    return x