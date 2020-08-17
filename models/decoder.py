'''
This code is to construct decoder for SAR - two layer LSTMs combined with feature map with attention mechanism 
'''
import torch
import torch.nn as nn

__all__ = ['word_embedding','attention','decoder']

class word_embedding(nn.Module):
    def __init__(self, output_classes, embedding_dim):
        super(word_embedding, self).__init__()
        '''
        output_classes: number of output classes for the one hot encoding of a word
        embedding_dim: embedding dimension for a word
        '''
        self.linear = nn.Linear(output_classes, embedding_dim) # linear transformation

    def forward(self,x):
        x = self.linear(x)

        return x

class attention(nn.Module):
    def __init__(self, hidden_units_encoder, embedding_dim, H, W, D):
        super(attention, self).__init__()
        '''
        hidden_units_encoder: hidden units of encoder
        embedding_dim: embedding dimension of a word
        H: height of feature map
        W: width of feature map
        D: depth of feature map
        '''
        self.conv1 = nn.Conv2d(hidden_units_encoder, D, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(D, 1, kernel_size=1, stride=1)
        self.softmax2d = nn.Softmax2d()
        self.H = H
        self.W = W
        self.D = D

    def forward(self, h, feature_map):
        '''
        h: hidden state from encoder output, with size [batch, hidden_units]
        feature_map: feature map from backbone network, with size [batch, channel, H, W]
        '''
        # reshape hidden state [batch, hidden_units] to [batch, hidden_units, 1, 1]
        h = h.unsqueeze(2)
        h = h.unsqueeze(3)
        h = self.conv1(h) # [batch, D, 1, 1]
        h = h.repeat(1, 1, self.H, self.W) # tiling to [batch, D, H, W]
        feature_map_origin = feature_map
        feature_map = self.conv2(feature_map) # [batch, D, H, W]
        combine = self.conv3(torch.tanh(feature_map + h)) # [batch, 1, H, W]
        attention_weights = self.softmax2d(combine) # [batch, 1, H, W]
        glimpse = feature_map_origin * attention_weights.repeat(1, self.D, 1, 1) # [batch, D, H, W]
        glimpse = torch.sum(glimpse, dim=(2,3)) # [batch, D]
        glimpse = glimpse.unsqueeze(2) # [batch, D, 1]
        glimpse = glimpse.unsqueeze(3) # [batch, D, 1, 1]

        return glimpse

class decoder(nn.Module):
    def __init__(self, output_classes, hidden_units=512, layers=2, embedding_dim=512, seq_len=40, lstm_keep_prob=1.0, att_keep_prob=1.0, training=True):
        super(decoder, self).__init__()
        pass

    def forward(self,x):
        pass

# unit test
if __name__ == '__main__':
    import torch

    batch_size = 2
    Height = 48
    Width = 160
    Channel = 512
    output_classes = 94
    embedding_dim = 512
    hidden_units_encoder = 512

    one_hot_embedding = torch.randn(batch_size, output_classes)
    one_hot_embedding[one_hot_embedding>0] = torch.ones(1)
    one_hot_embedding[one_hot_embedding<0] = torch.zeros(1)
    print("Word embedding size is:", one_hot_embedding.shape)

    embedding_model = word_embedding(output_classes, embedding_dim)
    embedding_transform = embedding_model(one_hot_embedding)
    print("Embedding transform size is:", embedding_transform.shape)

    hw = torch.randn(batch_size, hidden_units_encoder)
    feature_map = torch.randn(batch_size,Channel,Height,Width)
    print("Feature map size is:", feature_map.shape)

    attention_model = attention(hidden_units_encoder, embedding_dim, Height, Width, Channel)
    glimpse = attention_model(hw, feature_map)
    print("Glimpse size is:", glimpse.shape)