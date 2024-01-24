import torch
import torch_geometric
from torch_geometric.nn import global_mean_pool
from torch.nn.functional import cosine_similarity # pdist
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

def batched_pairwise_cosine_similarity(x):
    return cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)

def batched_similarity_matrix(graph_batch):
    x = graph_batch.x
    
    similarities_per_graph = []

    elements_uniques, compteurs = torch.unique(graph_batch.batch, return_counts=True)

    for graph,n_node in zip(elements_uniques, compteurs):
        mask = (graph_batch.batch == graph).nonzero(as_tuple=False).squeeze()

        if n_node == 1:
            graph_similarities = batched_pairwise_cosine_similarity(x[mask].unsqueeze(0))
        else :
            graph_similarities = batched_pairwise_cosine_similarity(x[mask])

        similarities_per_graph.append(graph_similarities)

    similarity_matrix = torch.block_diag(*similarities_per_graph)
    
    return similarity_matrix


def fusion(graph_batch, k, beta): 
    """
    returns the similarity matrix and the adjacency matrix of a graph (combining initial graph and knn graph)
    see Lin et al. (2023)

    Input :
     - graph_batch contains the following attributes : x, edge_index, batch
    Ouput : 
     - fusion_graph : torch_geometric.Data object representing A_F
                    with attributes x, edge_index, edge_attr, batch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

    x = graph_batch.x
    edge_index = graph_batch.edge_index
    batch = graph_batch.batch
    n = x.shape[0]

    A = torch_geometric.utils.to_dense_adj(edge_index,batch = batch)


    similarity_matrix = batched_similarity_matrix(graph_batch)

    # Step : knn graph
    _,nearest_neighbors = torch.topk(similarity_matrix, dim=1, largest=True, k=k+1)
    mask = (nearest_neighbors.unsqueeze(1) == nearest_neighbors.unsqueeze(0)).any(dim=-1).to(torch.int)
    mask = mask - torch.eye(n, dtype=torch.int, device = device)
    A_K = mask * similarity_matrix


    # passage de A_K (N,N) à (32,n_max,n_max)
    batch_size = batch.max().item() + 1
    n_max = A.shape[1]
    A_K_reshape = torch.zeros(batch_size, n_max, n_max, device = device)

    for batch_num in range(batch_size):
        batch_ind = torch.where(batch == batch_num*torch.ones(n, device = device))[0]

        begin = batch_ind.min().item()
        end = batch_ind.max().item()

        current_n_node = end + 1 - begin

        A_K_reshape[batch_num,:current_n_node,:current_n_node] = A_K[begin:end+1 , begin:end+1]


    # Step :  fusion
    

    A_F = beta * A + (1-beta)*A_K_reshape

    edge_indices = []
    edge_attrs = []

    for batch_idx in range(batch_size):
        edge_index = A_F[batch_idx].nonzero().t()

        edge_attr = A_F[batch_idx][edge_index[0], edge_index[1]].view(-1, 1)

        edge_indices.append(edge_index)
        edge_attrs.append(edge_attr)

    final_edge_index = torch.cat(edge_indices, dim=1)
    final_edge_attr = torch.cat(edge_attrs, dim=0)

    return similarity_matrix, (final_edge_index, final_edge_attr)



class GraphDecoder(nn.Module):

    def __init__(self, ):
        super(GraphDecoder, self).__init__()

    def forward(self,Z, batch):
        """
        Z : (Total number of nodes, ...)
        batch : (number of nodes) indique dans quel batch le noeud appartient

        """

        obj = torch_geometric.data.Data(x = Z, batch = batch)

        A = torch.sigmoid(Z @ Z.T)
        similarity_matrix = batched_similarity_matrix(obj)

        return A, similarity_matrix

def reconstructive_loss(A, batch, similarity_matrix, A_decoded, similarity_matrix_decoded, lambda_):
    """
    reconstructive loss (convex combination of structural and similarity loss) between
    - original graph : A, batch, similarity_matrix
    - decoded graph : A_decoded, similarity_matrix_decoded
    """

    strutural_loss = 0
    for batch_num in range(batch.batch.max().item()+1):
        batch_ind = torch.where(batch.batch == batch_num*torch.ones(batch.batch.shape[0], device = device))[0]
        begin = batch_ind.min().item()
        end = batch_ind.max().item()
        current_n_node = end + 1 - begin

        A_current_graph = A[batch_num,:current_n_node,:current_n_node]
        A_decoded_current_graph = A_decoded[begin:end+1 , begin:end+1]

        strutural_loss -= torch.mean(A_current_graph * torch.log(A_decoded_current_graph) + \
            (torch.ones(current_n_node)- A_current_graph) * torch.log(torch.ones(current_n_node)-A_decoded_current_graph))


    similarity_loss = torch.norm(similarity_matrix - similarity_matrix_decoded, p = 'fro')

    current_loss = lambda_ * strutural_loss + (1-lambda_) * similarity_loss

    return current_loss, strutural_loss, similarity_loss