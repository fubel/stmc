import rama_py
import torch


def multicut(edge_index, edge_weights, opts):
    """Solves a multicut problem based on the RAMA algorithm.

    The edge_index is expected in the usual torch_geometric format.
    Note that RAMA requires u < v for each edge (u, v) in the graph.

    Args:
        edge_index (LongTensor): 2xE LongTensor of edge indices.
        edge_weights (LongTensor): E LongTensor of edge weights.

    Returns:
        LongTensor: N LongTensor of node labels, where N is the number
            of nodes in the graph.
    """
    if (edge_index[0] > edge_index[1]).any():
        raise ValueError("Solver expects u < v for each edge (u, v) in the graph.")
    if edge_index.device.index is None:
        raise ValueError("Solver runs on CUDA device only. Please move data to CUDA.")
    if edge_index.shape[1] == 0:
        return torch.empty(0).to("cuda")
    i = edge_index[0].to(torch.int32)
    j = edge_index[1].to(torch.int32)
    costs = edge_weights.to(torch.float32)
    num_nodes = torch.max(edge_index) + 1
    num_edges = edge_index.shape[1]
    node_labels = torch.ones(num_nodes, device=i.device).to(torch.int32)
    rama_py.rama_cuda_gpu_pointers(
        i.data_ptr(),
        j.data_ptr(),
        costs.data_ptr(),
        node_labels.data_ptr(),
        num_nodes,
        num_edges,
        i.device.index,
        opts,
    )
    return node_labels


def scale_weights(weights, threshold=0.7):
    """Scales the given weights to the range [-1, 1] based on the given threshold.

    Args:
        weights (FloatTensor): LongTensor of edge weights.
        threshold (float, optional): Threshold for scaling. Defaults to 0.4.

    Returns:
        FloatTensor: LongTensor of scaled edge weights.
    """
    y = weights.clone()
    z = weights.clone()
    z[y == threshold] = 0.0
    z[y > threshold] = (y[y > threshold] - threshold) / (1 - threshold)
    z[y < threshold] = (y[y < threshold] - threshold) / (threshold)
    return z


def create_solver(backend):
    opts = rama_py.multicut_solver_options(backend)
    opts.verbose = False
    return opts
