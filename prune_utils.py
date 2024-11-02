import torch 
import torch.nn as nn 
from layerwrapper import WrappedLayer 

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    subset = find_layers(model, layers=[nn.Linear])
    zero_cnt = 0
    fc_params = 0
    for name in subset:
        W = subset[name].weight.data
        if W.shape[0] == 1000:
            continue 
        zero_cnt += (W==0).sum().item()
        fc_params += W.numel()
    return float(zero_cnt) / fc_params

def compute_mask(metric, granularity):
    """
    Compute pruning mask based on the metric to achieve 50% sparsity
    Args:
        metric: The metric to use for pruning (weight magnitude or WANDA score)
        granularity: 'per_layer' or 'per_channel'
    Returns:
        torch.Tensor: Boolean mask where True indicates weights to be pruned
    """
    # Ensure metric is a tensor
    if not isinstance(metric, torch.Tensor):
        metric = torch.tensor(metric)
    
    # Initialize mask
    mask = torch.zeros_like(metric, dtype=torch.bool, device=metric.device)
    
    if granularity == "per_layer":
        # Flatten the metric for layer-wise pruning
        metric_flat = metric.flatten()
        # Calculate threshold for 50% pruning
        k = len(metric_flat) // 2  # This ensures 50% sparsity
        if k < 1:
            return torch.ones_like(metric, dtype=torch.bool)
        threshold = torch.kthvalue(metric_flat, k).values
        # Create mask
        mask = (metric <= threshold).to(metric.device)
        
    elif granularity == "per_channel":
        # Apply 50% pruning per output channel
        for i in range(metric.shape[0]):
            channel_metric = metric[i].flatten()
            k = len(channel_metric) // 2  # 50% sparsity per channel
            if k < 1:
                mask[i] = torch.ones_like(metric[i], dtype=torch.bool)
                continue
            threshold = torch.kthvalue(channel_metric, k).values
            mask[i] = metric[i] <= threshold
    else:
        raise ValueError(f"Unsupported granularity: {granularity}. Choose 'per_layer' or 'per_channel'")
            
    return mask


def prune_deit(args, model, calib_data, device):
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda"])

    metric_stats = []
    for blk in model.blocks:
        subset = find_layers(blk)
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)

    thresh = None 
    #####################################
    inps = model.patch_embed(inps)

    cls_tokens = model.cls_token.expand(bs, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    dist_token = model.dist_token.expand(bs, -1, -1)
    inps = torch.cat((cls_tokens, dist_token, inps), dim=1)

    inps = inps + model.pos_embed
    inps = model.pos_drop(inps)

    for block_id, blk in enumerate(model.blocks):
        subset = find_layers(blk)

        if require_forward:
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            if bs > 256:
                tmp_res = []
                for i1 in range(0, bs, 256):
                    j1 = min(i1+256, bs)
                    tmp_res.append(blk(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = blk(inps)

            for h in handles:
                h.remove()

        ################# pruning ###################
        for name in subset:
            if args.prune_metric == "wanda":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, args.sparsity)

            subset[name].weight.data[W_mask] = 0

def prune_vit(args, model, calib_data, device):
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda"])

    metric_stats = []
    for blk in model.blocks:
        subset = find_layers(blk)
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)

    thresh = None 
    #####################################
    inps = model.patch_embed(inps)

    cls_tokens = model.cls_token.expand(bs, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    inps = torch.cat((cls_tokens, inps), dim=1)
    inps = inps + model.pos_embed
    inps = model.pos_drop(inps)

    for block_id, blk in enumerate(model.blocks):
        # print(f"block {block_id}")
        subset = find_layers(blk)

        if require_forward:
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            if bs > 256:
                tmp_res = []
                for i1 in range(0, bs, 256):
                    j1 = min(i1+256, bs)
                    tmp_res.append(blk(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = blk(inps)

            for h in handles:
                h.remove()

        ################# pruning ###################
        for name in subset:
            if args.prune_metric == "wanda":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, args.sparsity)

            subset[name].weight.data[W_mask] = 0
        ##############################################
import torch

def get_layer_sparsity(model):
    """Helper function to get detailed sparsity information"""
    total_weights = 0
    total_nonzero = 0
    layer_stats = {}
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight.data
            n_weights = weights.numel()
            n_nonzero = torch.count_nonzero(weights).item()
            sparsity = 1.0 - (n_nonzero / n_weights)
            
            total_weights += n_weights
            total_nonzero += n_nonzero
            layer_stats[name] = {
                'total': n_weights,
                'nonzero': n_nonzero,
                'sparsity': sparsity
            }
    
    return layer_stats, total_nonzero, total_weights
def prune_convnext(args, model, calib_data, device):
    """
    Guaranteed weight reduction pruning implementation
    """
    def count_nonzero_weights(model):
        return sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad)

    initial_nonzero = count_nonzero_weights(model)
    print(f"Initial non-zero weights: {initial_nonzero}")

    # Process each layer directly
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Get the weight tensor
            weight = module.weight.data
            
            # Count current non-zero weights
            mask = weight != 0
            n_nonzero = torch.count_nonzero(mask).item()
            
            if n_nonzero > 1:  # Only prune if there are weights to prune
                # Calculate absolute values for importance
                importance = torch.abs(weight)
                
                # Only consider non-zero weights
                importance[~mask] = 0
                
                # Flatten for easier processing
                flat_importance = importance.view(-1)
                flat_mask = mask.view(-1)
                
                # Get indices of non-zero weights
                nonzero_idx = torch.nonzero(flat_mask).squeeze()
                
                if len(nonzero_idx) > 0:
                    # Sort non-zero weights by importance
                    nonzero_importance = flat_importance[nonzero_idx]
                    sorted_idx = torch.argsort(nonzero_importance)
                    
                    # Calculate how many weights to prune (50% of current non-zero)
                    n_to_prune = max(len(nonzero_idx) // 2, 1)
                    
                    # Get indices to prune
                    prune_idx = nonzero_idx[sorted_idx[:n_to_prune]]
                    
                    # Create pruning mask
                    flat_weight = weight.view(-1)
                    flat_weight[prune_idx] = 0
                    
                    # Reshape back
                    weight.data = flat_weight.view(weight.shape)
                    
                    # Verify pruning for this layer
                    new_nonzero = torch.count_nonzero(weight).item()
                    print(f"\nLayer {name}:")
                    print(f"  Before pruning: {n_nonzero}")
                    print(f"  After pruning: {new_nonzero}")
                    print(f"  Weights pruned: {n_nonzero - new_nonzero}")

    # Verify overall pruning effectiveness
    final_nonzero = count_nonzero_weights(model)
    weights_pruned = initial_nonzero - final_nonzero
    print(f"\nOverall pruning results:")
    print(f"Total weights pruned: {weights_pruned}")
    print(f"New total non-zero weights: {final_nonzero}")
    
    if final_nonzero >= initial_nonzero:
        raise ValueError("Pruning failed to reduce weights!")
        
    return model
# def compute_mask(metric, granularity, sparsity):
#     # Add your mask computation logic here
#     # For demonstration purposes, we'll assume this function returns a valid mask or None
#     # Ensure to check for sparsity and granularity in your implementation.
#     if metric is None or metric.numel() == 0:
#         return None  # Return None for invalid metrics
#     # Example mask computation logic (this should be tailored to your use case)
#     threshold = torch.quantile(metric, sparsity)
#     W_mask = metric < threshold
#     return W_mask

def check_sparsity(model):
    total_params = 0
    total_sparsity = 0
    for param in model.parameters():
        total_params += param.numel()
        total_sparsity += torch.sum(param == 0).item()
    return total_sparsity / total_params

# Usage example (not complete, as it requires the actual model and args)
# args = ...  # Set your args here
# model = ...  # Your ConvNext model here
# calib_data = ...  # Your calibration data here
# device = ...  # Your device (e.g., 'cuda' or 'cpu')
# with torch.no_grad():
#     if "convnext" in args.model:
#         prune_convnext(args, model, calib_data, device)
#     actual_sparsity = check_sparsity(model)
#     print(f"Actual sparsity after pruning: {actual_sparsity}")
