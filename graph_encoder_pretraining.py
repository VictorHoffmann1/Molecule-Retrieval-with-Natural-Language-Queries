from dataloader import GraphTextDataset, GraphDataset, TextDataset
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from Model import GraphEncoder
import numpy as np
import torch
from torch import nn
from torch import optim
import time
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv



#define the decoder and the AE using to pretrain the graphEncoder
class GraphDecoder(nn.Module):
    """
    reconstruct each node using the compressed representation of graph at node level
    use only to pretrain the GraphEncoder
    """
    def __init__(self, latent_size, num_heads, output_size=300):
      """
      Only 1 layer because it is the encoder that is important
      """
      super(GraphDecoder, self).__init__()
      self.gat1 = GATConv(latent_size*num_heads, output_size, heads=1)

    def forward(self, z, edge_index):
        x_reconstructed = self.gat1(z, edge_index)
        return x_reconstructed


class Graph_AE(nn.Module):
    """
    use to pretrain a part of the graph encoder (the 3 GATs)
    """
    def __init__(self, input_size=300, n_out=32, latent_size=64, num_heads=2):
        super(Graph_AE, self).__init__()
        self.encoder = GraphEncoder(input_size, n_out, latent_size, num_heads) #the one use in our model
        self.decoder = GraphDecoder(latent_size, num_heads, input_size)

    def forward(self, x, edge_index):
        # load only the node level part of the GraphEncoder (GAT layers)
        x = self.encoder.conv1(x, edge_index)
        x = self.encoder.relu(x)
        x = self.encoder.dropout(x)
        x = self.encoder.conv2(x, edge_index)
        x = self.encoder.relu(x)
        x = self.encoder.dropout(x)
        x = self.encoder.conv3(x, edge_index)
        z = self.encoder.relu(x) 
        #reconstruct the nodes using the decoder
        x_recon = self.decoder(z, edge_index)
        return x_recon
    
## Load the data
batch_size = 64
tokenizer_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

##Parameters
input_size = 300
n_out = 32
latent_size = 300
num_heads = 4
batch_size = 64


#train the model
model = Graph_AE(input_size, n_out, latent_size, num_heads)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 50


train_loss_tracking=[]
val_loss_tracking=[]

for epoch in range(0, num_epochs):
    print('-----EPOCH{}-----'.format(epoch+1))
    total_loss = 0
    for batch in train_loader:
        model.train()
        optimizer.zero_grad()

        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch

        x = graph_batch.x.to(device)
        edge_index = graph_batch.edge_index.to(device)

        x_recon = model(x, edge_index)
        loss = criterion(x_recon, x)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    train_loss_tracking.append(average_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')
    model.eval()
    val_loss = 0
    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch

        x = graph_batch.x.to(device)
        edge_index = graph_batch.edge_index.to(device)

        x_recon = model(x, edge_index)
        current_loss = criterion(x_recon, x)
        val_loss += current_loss.item()

    average_loss = val_loss / len(val_loader)
    val_loss_tracking.append(average_loss)
    print('-----EPOCH'+str(epoch+1)+'----- done.  Validation loss: ', f'{average_loss}' )

#Save the encoder weights
torch.save(model.encoder.state_dict(), 'pretraining_GraphEncoder_50epochs.pt')

epochs = range(1, len(train_loss_tracking) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_tracking, label='Train Loss', marker='o')
plt.plot(epochs, val_loss_tracking, label='Validation Loss', marker='o')

plt.title('Training Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()