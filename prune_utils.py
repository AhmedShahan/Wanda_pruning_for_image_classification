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
def prune_convnext(args, model, calib_data, device):
    """
    Prune ConvNeXt model with 50% sparsity per layer of remaining non-zero weights
    """
    inps = calib_data
    bs = inps.shape[0]
    
    total_params = 0
    pruned_params = 0
    
    for block_id in range(4):
        print(f"Processing block {block_id}")
        subset = find_layers(model.stages[block_id])
        
        # Forward pass for WANDA metric calculation
        layer = model.downsample_layers[block_id]
        if bs > 1024:
            tmp_res = []
            for i1 in range(0, bs, 512):
                j1 = min(i1+512, bs)
                tmp_res.append(layer(inps[i1:j1]))
            inps = torch.cat(tmp_res, dim=0)
        else:
            inps = layer(inps)

        # Wrap layers and collect statistics
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedLayer(subset[name])
            
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(
                lambda n, i, o, name=name: wrapped_layers[name].add_batch(i[0].data, o.data)
            ))
            
        # Forward pass through the block
        layer = model.stages[block_id]
        if bs > 1024:
            tmp_res = []
            for i1 in range(0, bs, 512):
                j1 = min(i1+512, bs)
                tmp_res.append(layer(inps[i1:j1]))
            inps = torch.cat(tmp_res, dim=0)
        else:
            inps = layer(inps)
            
        for h in handles:
            h.remove()

        # Modified pruning section
        for name in subset:
            weight = subset[name].weight.data
            total_params += weight.numel()
            
            # Get current non-zero weights
            current_nonzero = weight != 0
            nonzero_count = current_nonzero.sum().item()
            
            # Calculate target number of weights to keep (50% of current non-zero)
            target_nonzero = nonzero_count // 2
            
            # Calculate WANDA metric only for non-zero weights
            metric = torch.abs(weight) * current_nonzero.float()
            if args.prune_metric == "wanda":
                scaler = wrapped_layers[name].scaler_row
                if scaler is not None:
                    metric *= torch.sqrt(scaler.reshape((1, -1)))
            
            # Find threshold for keeping top 50% of non-zero weights
            if nonzero_count > 0:
                # Get values only from non-zero positions
                nonzero_metrics = metric[current_nonzero]
                if len(nonzero_metrics) > 0:
                    threshold = torch.sort(nonzero_metrics)[0][target_nonzero]
                    mask = metric <= threshold
                    
                    # Only apply mask to non-zero weights
                    mask = mask & current_nonzero
                    
                    # Apply pruning
                    subset[name].weight.data[mask] = 0
                    
                    # Track pruned parameters
                    pruned_params += mask.sum().item()
                    
                    # Calculate and print layer sparsity
                    layer_sparsity = (weight == 0).sum().item() / weight.numel()
                    print(f"Layer {name} sparsity: {layer_sparsity:.4f}")
                    print(f"Layer {name} non-zero weights remaining: {(weight != 0).sum().item()}")
    
    overall_sparsity = pruned_params / total_params if total_params > 0 else 0
    print(f"Overall model sparsity: {overall_sparsity:.4f}")
    
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
