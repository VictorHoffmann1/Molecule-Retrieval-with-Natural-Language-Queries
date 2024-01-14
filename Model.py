import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
import math

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, num_heads):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        
        # Replace GCNConv with GATConv
        self.conv1 = GATConv(num_node_features, graph_hidden_channels, heads=num_heads)
        self.conv2 = GATConv(graph_hidden_channels * num_heads, graph_hidden_channels, heads=num_heads)
        self.conv3 = GATConv(graph_hidden_channels * num_heads, graph_hidden_channels, heads=num_heads)
        
        self.mol_hidden1 = nn.Linear(graph_hidden_channels * num_heads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        # Replace GCN layers with GAT layers
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, nhid, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, nhid)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, nhid, 2).float() * (-math.log(10000.0) / nhid)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class TextEncoder(nn.Module):
    def __init__(self, ntoken, nhead, nhid, nlayers, dropout=0.3):
        super(TextEncoder, self).__init__()
        '''
        ntokens: the size of vocabulary
        nhid: the hidden dimension of the model.
        We assume that embedding_dim = nhid
        nlayers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead: the number of heads in the multiheadattention models
        dropout: the dropout value
         '''
        self.model_type = "Transformer"
        self.encoder = nn.Embedding(ntoken, nhid)
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        encoder_layers = nn.TransformerEncoderLayer(nhid, nhead, dropout = dropout, dim_feedforward = nhid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.nhid = nhid
        self.init_weights()

        self.weighted_mean = nn.Parameter(torch.ones(256))
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        
        src = self.encoder(src) * math.sqrt(self.nhid)
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, src_mask, src_key_padding_mask = src_key_padding_mask)
        attention_weights = F.softmax(src, dim=1)  # Apply softmax along sentence_length dimension

        src = src.transpose(1, 2)
        output = src @ self.weighted_mean

        return output

class Model(nn.Module):
    def __init__(self, num_node_features, nhid_gat, graph_hidden_channels, num_head_gat, ntoken, num_head_text, nhid_text, nlayers_text, dropout=0.3):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nhid_text, nhid_gat, graph_hidden_channels, num_head_gat)
        self.text_encoder = TextEncoder(ntoken, num_head_text, nhid_text, nlayers_text, dropout)
        
    def forward(self, graph_batch, input_ids, attention_mask, src_key_padding_mask = None):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask, src_key_padding_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
