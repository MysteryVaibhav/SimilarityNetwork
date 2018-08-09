import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityNetwork(torch.nn.Module):
    def __init__(self, params):
        super(SimilarityNetwork, self).__init__()
        self.params = params
        self.word_embeddings = nn.Embedding(params.vocab_size, params.embedding_dimension)
        nn.init.xavier_uniform(self.word_embeddings.weight)
        if params.use_conv_encoder == 0:
            self.text_encoder = Encoder(params.hidden_dimension, params.embedding_dimension)
        else:
            self.text_encoder = ConvEncoder(params.hidden_dimension, params.embedding_dimension)
        self.image_branch = Branch(params.visual_feature_dimension, params.hidden_dimension, params.dropout)
        self.text_branch = Branch(params.hidden_dimension, params.hidden_dimension, params.dropout)
        self.fc1 = nn.Linear(params.hidden_dimension, params.hidden_dimension)
        self.fc2 = nn.Linear(params.hidden_dimension, params.hidden_dimension // 2)
        self.fc3 = nn.Linear(params.hidden_dimension // 2, 1)
        self.non_linear = nn.ReLU()

    def forward(self, input_caption, mask, input_image, is_inference, is_phrase=False):
        # Embed sentences for text encoding
        embeds = self.word_embeddings(input_caption)                           # bs * max_seq_len * embedding_dimension
        h_t = self.text_encoder(embeds, mask)                                  # bs * hidden_dimension

        # Image branch
        embed_i = self.image_branch(input_image)                               # bs * hidden_dimension

        # Text branch
        embed_t = self.text_branch(h_t)                                        # bs * hidden_dimension

        # Take element-wise dot product
        dot = embed_i * embed_t                                                # bs * hidden_dimension

        x = self.fc1(dot)
        x = self.non_linear(x)
        x = self.fc2(x)
        x = self.non_linear(x)
        x = self.fc3(x)
        x = self.non_linear(x)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, hidden_dimension, embedding_dimension):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dimension
        self.gru = nn.GRU(embedding_dimension, hidden_dimension, bidirectional=True)

    def forward(self, embeds, mask):
        # Sorting sequences by their lengths for packing
        embeds = embeds.permute(1, 0, 2)                                # seq_len * batch_size * embedding_dimension

        h, _ = self.gru(embeds)

        # h = ( h_forward + h_backward ) / 2
        h = h.view(h.size(0), h.size(1), 2, -1).sum(2) / 2

        h = h.permute(1, 0, 2)                                       # bs * max_seq_len * hidden_dim
        h = h.sum(dim=1) / h.size(1)
        return h

    
class ConvEncoder(torch.nn.Module):
    def __init__(self, hidden_dimension, embedding_dimension):
        super(ConvEncoder, self).__init__()
        self.hidden_dim = hidden_dimension
        self.conv_unigram = nn.Conv1d(in_channels=embedding_dimension, out_channels=hidden_dimension, kernel_size=1, padding=0)
        self.conv_bigram = nn.Conv1d(in_channels=embedding_dimension, out_channels=hidden_dimension, kernel_size=5, padding=2)
        self.conv_trigram = nn.Conv1d(in_channels=embedding_dimension, out_channels=hidden_dimension, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=3 * hidden_dimension, out_features=hidden_dimension)

    def forward(self, embeds, mask):
        embeds = embeds.permute(0, 2, 1)                                                       # bs * ed * seq
        h_uni = self.conv_unigram(embeds)                                                      # bs * hd * seq
        h_bi = self.conv_bigram(embeds)                                                        # bs * hd * seq
        h_tri = self.conv_trigram(embeds)                                                      # bs * hd * seq
        h = torch.cat((h_uni, h_bi, h_tri), dim=1)
        h = h.permute(0, 2, 1)
        h = self.fc(h)
        h = h.sum(dim=1) / h.size(1)
        return h


class Branch(torch.nn.Module):
    def __init__(self, input_dim, hidden_dimension, dropout):
        super(Branch, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=2 * hidden_dimension)
        self.fc2 = nn.Linear(in_features=2 * hidden_dimension, out_features=hidden_dimension)
        self.b_norm = nn.BatchNorm1d(num_features=2 * hidden_dimension, momentum=0.1, eps=1e-5)
        self.non_linear = nn.ReLU()
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.b_norm(x)
        x = self.non_linear(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-10)
        return x