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

def compute_mask(W_metric, prune_granularity, sparsity):
    if prune_granularity == "layer":
        thres = torch.sort(W_metric.flatten().cuda())[0][int(W_metric.numel() * sparsity)].cpu()
        W_mask = (W_metric <= thres)
        return W_mask 
    elif prune_granularity == "row":
        W_mask = (torch.zeros_like(W_metric)==1)
        sort_res = torch.sort(W_metric, dim=-1, stable=True)

        indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity)]
        W_mask.scatter_(1, indices, True)
        return W_mask 

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
        print(f"block {block_id}")
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
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda"])

    ##############################################################
    metric_stats = []
    for block_id in range(4):
        subset = find_layers(model.stages[block_id])
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)
    ##############################################################

    thresh = None 
    for block_id in range(4):
        print(f"Block {block_id}")
        subset = find_layers(model.stages[block_id])

        if require_forward:
            layer = model.downsample_layers[block_id]
            if bs > 1024:
                tmp_res = []
                for i1 in range(0, bs, 512):
                    j1 = min(i1 + 512, bs)
                    tmp_res.append(layer(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = layer(inps)

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
            layer = model.stages[block_id]
            if bs > 1024:
                tmp_res = []
                for i1 in range(0, bs, 512):
                    j1 = min(i1 + 512, bs)
                    tmp_res.append(layer(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = layer(inps)
            for h in handles:
                h.remove()

        ################# Pruning ###################
        for name in subset:
            if args.prune_metric == "wanda":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            # Debugging: Check the metric before computing the mask
            print(f"Layer: {name}, Metric before mask: {metric_stats[block_id][name]}")

            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, args.sparsity)

            if W_mask is None:
                print(f"Warning: W_mask is None for {name}. Skipping pruning for this layer.")
                continue  # Skip this layer or handle it accordingly

            # Debugging: Check the mask and weights
            print(f"W_mask for {name}: {W_mask}, Shape: {W_mask.shape}")
            print(f"Weight shape for {name}: {subset[name].weight.data.shape}")

            subset[name].weight.data[W_mask] = 0

            # Debugging: Check the number of non-zero weights after pruning
            print(f"Non-zero weights for {name} after pruning: {torch.count_nonzero(subset[name].weight.data)}")
        ##############################################

def compute_mask(metric, granularity, sparsity):
    # Add your mask computation logic here
    # For demonstration purposes, we'll assume this function returns a valid mask or None
    # Ensure to check for sparsity and granularity in your implementation.
    if metric is None or metric.numel() == 0:
        return None  # Return None for invalid metrics
    # Example mask computation logic (this should be tailored to your use case)
    threshold = torch.quantile(metric, sparsity)
    W_mask = metric < threshold
    return W_mask

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
