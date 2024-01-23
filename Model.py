import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import math
from transformers import AutoModel
from torch_geometric.nn import GATConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, heads=1):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.heads = heads
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.ln = nn.LayerNorm((nout))

        # Define the GAT layers
        self.conv1 = GATConv(num_node_features, nhid, heads=self.heads)
        self.conv2 = GATConv(nhid * self.heads, nhid, heads=self.heads)
        self.conv3 = GATConv(nhid * self.heads, nhid, heads=self.heads)
        self.mol_hidden1 = nn.Linear(nhid * self.heads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch

        # Apply the GAT layers
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)       

        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x



class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.model_type = "Transformer"
        
        # Use pretrained BERT model as encoder
        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.nhid = self.encoder.config.hidden_size
        encoder_layers = nn.TransformerEncoderLayer(self.nhid, 4, dropout = 0.3, dim_feedforward = self.nhid)
        self.intermediate = nn.TransformerEncoder(encoder_layers, 2)
        # Define MLP for learning weights
        self.mlp = nn.Sequential(
            nn.Linear(self.nhid, self.nhid),
            nn.ReLU(),
            nn.Linear(self.nhid, 1),
            nn.Softmax(dim=1)
        )
        
        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def forward(self, src, padding_mask):
        # Pass input through BERT model
        src = self.encoder(src, attention_mask = padding_mask)
        src = self.intermediate(src.last_hidden_state)

        # Use MLP to compute weights
        weights = self.mlp(src)
        weights = weights.transpose(-2, -1)  # Ensure weights have the same shape as src

        # Compute weighted sum
        output = torch.bmm(weights, src).squeeze(-2)  # Use batch matrix-matrix product to compute weighted sum
        return output
    

class Model(nn.Module):
    
    def __init__(self, num_node_features, nhid_gat, num_head_gat):
        super(Model, self).__init__()
        self.text_encoder = TextEncoder()
        self.graph_encoder = GraphEncoder(num_node_features, self.text_encoder.nhid, nhid_gat, num_head_gat)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder