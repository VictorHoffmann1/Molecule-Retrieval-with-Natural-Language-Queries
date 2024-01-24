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

from GAE_tools import fusion, GraphDecoder, reconstructive_loss
from Model import GraphEncoder


### Data loading ###
batch_size = 32

tokenizer_name = 'alvaroalon2/biobert_chemical_ner'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


### Instanciation des modèles ###
learning_rate = 2e-5


model = Model(num_node_features=300, nhid_gat=300, graph_hidden_channels=300, num_head_gat=8, 
    ntoken=tokenizer.vocab_size, num_head_text=8, nhid_text=512, nlayers_text=8, dropout=0.3,
    fusion_k = 5, fusion_beta = 1)
model.to(device)

graph_encoder = model.get_graph_encoder()
graph_decoder = GraphDecoder()

optimizer = optim.AdamW(model.graph_encoder.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)

nb_epochs = 40

lambda_ = 0.5
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000



for i in range(nb_epochs):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()

    for batch in train_loader:
        input_ids = batch.input_ids

        similarity_matrix, (edge_index, edge_attr) = fusion(batch.to(device), k = 1, beta=0.2)
        graphFusion_batch = torch_geometric.data.Data(x = batch.x, edge_index = edge_index, edge_attr = edge_attr, batch = batch.batch)

        Z = graph_encoder(graphFusion_batch, graph_pretraining = True)
        A_decoded, similarity_matrix_decoded = graph_decoder(Z, batch.batch)

        A = torch_geometric.utils.to_dense_adj(batch.edge_index, batch = batch.batch)

        current_loss,_,_ = reconstructive_loss(A,batch,similarity_matrix,A_decoded,similarity_matrix_decoded,lambda_)

        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()

        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter, time2 - time1, loss/printEvery))
            losses.append(loss)
            loss = 0 


    model.eval()
    val_loss = 0        
    for batch in val_loader:
        batch.pop('input_ids')
        batch.pop('attention_mask')


        similarity_matrix, (edge_index, edge_attr) = fusion(batch.to(device), k = 1, beta=0.2)
        graphFusion_batch = torch_geometric.data.Data(x = batch.x, edge_index = edge_index, edge_attr = edge_attr, batch = batch.batch)

        Z = graph_encoder(graphFusion_batch, graph_pretraining = True)
        A_decoded, similarity_matrix_decoded = graph_decoder(Z, batch.batch)

        A = torch_geometric.utils.to_dense_adj(batch.edge_index, batch = batch.batch)

        current_loss, _, _ = reconstructive_loss(A,batch,similarity_matrix,A_decoded,similarity_matrix_decoded,lambda_)

        val_loss += current_loss.item()

    
    best_validation_loss = min(best_validation_loss, val_loss)
    
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )

    if best_validation_loss==val_loss:
        print('validation loss improved saving checkpoint...')
        save_path = os.path.join('./', 'graph_pretrained_model'+str(i)+'.pt')
        torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'validation_accuracy': val_loss
        }, save_path)
        print('checkpoint saved to: {}'.format(save_path))
    
        
       