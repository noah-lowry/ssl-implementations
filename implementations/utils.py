import torch


def infonce(queries: torch.Tensor, positive_keys: torch.Tensor, negative_keys: torch.Tensor, tau: float = 0.07):
    """
    InfoNCE is the standard loss function of MoCo (tau=0.07 in v1, tau=0.2 in v2).\n

    ### Parameters
    - Let N be the number of positive keys (MoCo batch size), K be the number of negative keys (MoCo queue length), and C be the embedding dimensionality.\n
    - queries and positive_keys: float tensors both of shape (N, C) where (queries[i], positive_keys[i]) is a positive pair.\n
    - negative_keys: float tensor of shape (K, C) where (queries[i], negative_keys[j]) is a negative pair for all (i, j).\n
    - tau: temperature parameter (default 0.07).\n

    ### Returns
    The mean InfoNCE loss computed for all queries.
    """
    l_pos = torch.bmm(queries.unsqueeze(1), positive_keys.unsqueeze(2)).squeeze(1)
    l_neg = torch.mm(queries, negative_keys.T)
    logits = torch.cat([l_pos, l_neg], dim=1)
    labels = torch.zeros(queries.shape[0], device=logits.device, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits/tau, labels)
    return loss

def generalized_ntxent(tensor: torch.Tensor, pair_matrix: torch.Tensor, tau: float = 0.5):
    """
    NT-Xent is the standard loss function of SimCLR. The official implementation uses tau=0.5, although this is almost certainly not optimal (see paper).\n

    ### Parameters
    - tensor: float tensor of shape (N, C) where N is the batch size and C is the embedding dimensionality.\n
    - pair_matrix: float tensor or bool tensor of shape (N, N):
        - pair_matrix[i, j] is [True / greater that zero] iff (tensor[i], tensor[j]) is a positive pair of embeddings.
        - if pair_matrix has dtype float then the loss is computed as a weighted mean of positive pairs according to pair_matrix.\n
    - tau: temperature parameter (default 0.5).\n

    ### Returns
    The mean NT-Xent loss computed between all ordered* positive pairs.\n
    *Implying that (i, j) and (j, i) are considered distinct pairs (as in the paper).

    TODO: optimize this piece of shit function
    """
    assert tensor.ndim == 2

    result = torch.mm(tensor, tensor.T)
    result = torch.exp(result / (torch.outer(result.diag(), result.diag()).sqrt() * tau))  # 'result' currently holds the exponentiated similarity matrix

    result = -torch.log(result / torch.sum(result * (1-torch.eye(len(tensor))).to(result.device), dim=1, keepdim=True))

    if pair_matrix.dtype == torch.bool:
        return result[pair_matrix].mean()
    else:
        return torch.sum(result * pair_matrix) / torch.sum(pair_matrix)
