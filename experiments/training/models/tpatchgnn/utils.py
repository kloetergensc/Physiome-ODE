import torch


def split_and_patch_batch(batch, args):
    T, X, M, TY, Y, MY = (tensor.to(args.device) for tensor in batch)

    patch_indices = []
    st, ed = 0, args.patch_size
    for i in range(args.npatch):
        if i == args.npatch - 1:
            inds = torch.where((T >= st) & (T <= ed))[0]
        else:
            inds = torch.where((T >= st) & (T < ed))[0]
        patch_indices.append(inds)
        st += args.stride
        ed += args.stride
    device = args.device

    data_dict = {
        "tp_to_predict": TY,
        "data_to_predict": Y,
        "mask_predicted_data": MY,
        "time_steps": T,
        "data": X,
        "mask": M,
    }

    split_dict = {
        "tp_to_predict": data_dict["tp_to_predict"].clone(),
        "data_to_predict": data_dict["data_to_predict"].clone(),
        "mask_predicted_data": data_dict["mask_predicted_data"].clone(),
    }

    observed_tp = data_dict["time_steps"].clone().to(device)  # (n_observed_tp, )
    observed_data = data_dict["data"].clone()  # (bs, n_observed_tp, D)
    observed_mask = data_dict["mask"].clone()  # (bs, n_observed_tp, D)

    n_batch, n_tp, n_dim = observed_data.shape
    observed_tp_patches = observed_tp.view(1, 1, -1, 1).repeat(
        n_batch, args.npatch, 1, n_dim
    )
    observed_data_patches = observed_data.view(n_batch, 1, n_tp, n_dim).repeat(
        1, args.npatch, 1, 1
    )
    observed_mask_patches = observed_mask.view(n_batch, 1, n_tp, n_dim).repeat(
        1, args.npatch, 1, 1
    )

    max_patch_len = 0
    for i in range(args.npatch):
        indices = patch_indices[i]
        if len(indices) == 0:
            continue
        st_ind, ed_ind = indices[0], indices[-1]
        n_data_points = observed_mask[:, st_ind : ed_ind + 1].sum(dim=1).max().item()
        max_patch_len = max(max_patch_len, int(n_data_points))

    observed_mask_patches_fill = torch.zeros_like(
        observed_mask_patches, dtype=observed_mask.dtype
    )  # n_batch, npacth, n_tp, n_dim
    patch_indices_fianl = torch.full(
        (n_batch, args.npatch, max_patch_len, n_dim), n_tp
    ).to(
        device
    )  # n_batch, npacth, max_patch_len, n_dim
    observed_mask_patches_fill_reindex = torch.zeros_like(
        patch_indices_fianl, dtype=observed_mask.dtype
    )
    aux_tensor = (
        torch.arange(max_patch_len)
        .view(1, max_patch_len, 1)
        .repeat(n_batch, 1, n_dim)
        .to(device)
    )
    for i in range(args.npatch):
        indices = patch_indices[i]
        if len(indices) == 0:
            continue
        st_ind, ed_ind = indices[0], indices[-1]
        observed_mask_patches_fill[:, i, st_ind : ed_ind + 1] = observed_mask[
            :, st_ind : ed_ind + 1, :
        ]
        L = observed_mask[:, st_ind : ed_ind + 1, :].sum(
            dim=1, keepdim=True
        )  # (bs, 1, D)
        observed_mask_patches_fill_reindex[:, i] = (
            aux_tensor < L
        )  # let first L[i] to be True

    ### return a indices tuple like ([...], [...], [...], [...])
    mask_inds = torch.nonzero(
        observed_mask_patches_fill_reindex.permute(0, 1, 3, 2), as_tuple=True
    )  # reset indices
    ind_values = torch.nonzero(
        observed_mask_patches_fill.permute(0, 1, 3, 2), as_tuple=True
    )[
        -1
    ]  # original indices of dimension 2

    ### fill n_tp if the number of observed points are less than max_patch_len
    patch_indices_fianl.index_put_(
        (mask_inds[0], mask_inds[1], mask_inds[3], mask_inds[2]), ind_values
    )

    pad_zeros_data = torch.zeros([n_batch, args.npatch, 1, n_dim]).to(device)
    observed_tp_patches = torch.cat(
        [observed_tp_patches, pad_zeros_data], dim=2
    ).gather(
        2, patch_indices_fianl
    )  # (n_batch, npatch, max_patch_len, n_dim)
    observed_data_patches = torch.cat(
        [observed_data_patches, pad_zeros_data], dim=2
    ).gather(2, patch_indices_fianl)
    observed_mask_patches = torch.cat(
        [observed_mask_patches, pad_zeros_data], dim=2
    ).gather(2, patch_indices_fianl)

    split_dict["observed_tp"] = observed_tp_patches
    split_dict["observed_data"] = observed_data_patches
    split_dict["observed_mask"] = observed_mask_patches

    return split_dict
