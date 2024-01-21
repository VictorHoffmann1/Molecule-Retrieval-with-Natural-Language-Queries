from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import Model
import numpy as np
import torch
from torch import optim
import time
import os
import pandas as pd
from transformers import AutoTokenizer

from torch import nn
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
import math

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, num_heads):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.n_node_max = n_node_max
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

# class GraphDecoder(nn.Module):
#     def __init__(self, n_latent, n_hidden,n_layers, n_node_max,dropout = 0.2):

#         self.n_node_max = n_node_max

#         self.fc = nn.ModuleList()
#         self.fc.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim),  
#                             nn.ReLU(),
#                             nn.LayerNorm(hidden_dim), 
#                             nn.Dropout(dropout)
#                             ))

#         for i in range(1, n_layers):
#             self.fc.append(nn.Sequential(nn.Linear(hidden_dim*i, hidden_dim*(i+1)),  
#                             nn.ReLU(),
#                             nn.LayerNorm(hidden_dim*(i+1)), 
#                             nn.Dropout(dropout)
#                             ))

#         self.fc.append(nn.Sequential(nn.Linear(hidden_dim * n_layers,300 *)
#             ))

#     def forward(self, x): 



#         for i in range(self.n_layers):
#             x = self.fc[i](x)

#         Y = self.
#         # x = self.fc_proj(x)
#         # adj = x.reshape(-1 , self.n_nodes, self.n_nodes)
#         # adj = (adj + torch.transpose(adj, 1,2))/2

#         Y = 

#         return Y


n_node_max = 100 # ------------------------------------------ to be determined ---------------------------------------------------------------

num_node_features = 300
nout = 512 #nhid_text
nhid = 300
graph_hidden_channels = 64
num_heads = 4

graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels, num_heads)
#graph_decoder = GraphDecoder(n_latent = nout, n_hidden,n_layers, n_node_max, dropout = 0.2)

tokenizer_name = 'alvaroalon2/biobert_chemical_ner'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

nb_epochs = 40
batch_size = 32
learning_rate = 20e-5

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Model(num_node_features=300, nhid_gat=300, graph_hidden_channels=300, num_head_gat=8, ntoken=tokenizer.vocab_size, num_head_text=8, nhid_text=512, nlayers_text=8, dropout=0.3)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)

loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000

g_encoder = GraphEncoder(num_node_features = num_node_features, nout = nout, nhid = nhid, graph_hidden_channels= graph_hidden_channels, num_heads = num_heads)

from GAE_tools import fusion

for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()

    for batch in train_loader:
        input_ids = batch.input_ids

        attn_mask = model.text_encoder.generate_square_subsequent_mask(input_ids.size(0)).to(device)
        graph_batch = batch

        # print("graph batch", graph_batch)
        # print("x ", batch.x.shape)
        # print("statistiques sur x", torch.mean(batch.x, axis = 0), "std : ", torch.std(batch.x, axis = 0))
        # print(".batch",batch.batch)


        #S, A_F = FusionGraph(A, X, k, beta)

        #x_graph = batched_similarity_matrix(graph_batch.to(device))

        similarity_matrix, (edge_index, edge_attr) = fusion(graph_batch.to(device), k = 1, beta=0.2)
        graphFusion_batch = torch_geometric.data.Data(x = batch.x, edge_index = edge_index, edge_attr = edge_attr, batch = batch.batch)
        
        print(graphFusion_batch)
        print(graph_encoder(graphFusion_batch))

        # optimizer.zero_grad()
        # current_loss.backward()
        # optimizer.step()
        # loss += current_loss.item()

        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter, time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0 


    model.eval()       
    val_loss = 0        
    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attn_mask = model.text_encoder.generate_square_subsequent_mask(input_ids.size(0)).to(device)
        batch.pop('attention_mask')
        graph_batch = batch
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attn_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)   
        val_loss += current_loss.item()
    
    best_validation_loss = min(best_validation_loss, val_loss)
    
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )
    
        
       