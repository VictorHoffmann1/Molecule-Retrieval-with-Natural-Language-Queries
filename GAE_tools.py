import torch
from torch_geometric.nn import global_mean_pool
from torch.nn.functional import cosine_similarity # pdist


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



