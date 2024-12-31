import torch
from torch.optim.optimizer import Optimizer, required


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
    - tensor: float tensor of shape (N, C) where N is twice the batch size and C is the embedding dimensionality.\n
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


class LARS(torch.optim.Optimizer):
    
    def __init__(
        self,
        params,
        lr=required,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.eps = eps
        self.trust_coefficient = trust_coefficient

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0 and p_norm != 0 and g_norm != 0:
                    lars_lr = p_norm / (g_norm + p_norm * weight_decay + self.eps)
                    lars_lr *= self.trust_coefficient

                    d_p = d_p.add(p, alpha=weight_decay)
                    d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = d_p.add(buf, alpha=momentum) if nesterov else buf

                p.add_(d_p, alpha=-group["lr"])

        return loss
