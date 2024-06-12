def tsdm_sparse_collate(batch: list[Sample], sparsity: float) -> Batch:
    r"""Collate tensors into batch.

    Transform the data slightly: t, x, t_target → T, X where X[t_target:] = NAN
    """
    x_vals: list[Tensor] = []
    y_vals: list[Tensor] = []
    x_time: list[Tensor] = []
    y_time: list[Tensor] = []
    x_mask: list[Tensor] = []
    y_mask: list[Tensor] = []

    context_x: list[Tensor] = []
    context_vals: list[Tensor] = []
    context_mask: list[Tensor] = []
    target_vals: list[Tensor] = []
    target_mask: list[Tensor] = []

    for sample in batch:
        t, x, t_target = sample.inputs
        y = sample.targets
        # get whole time interval
        # create a mask for looking up the target values
        mask_y = y.isfinite()
        mask_x = x.isfinite()
        
        mask_x_inds = torch.where(mask_x.sum(-1).bool())
        mask_y_inds = torch.where(mask_y.sum(-1).bool())

        mask_x = mask_x[mask_x_inds]
        mask_y = mask_y[mask_y_inds]
        x = x[mask_x_inds]
        y = y[mask_y_inds]
        t = t[mask_x_inds]
        t_target = t_target[mask_y_inds]

        sorted_idx = torch.argsort(t)

        mask_y_float = mask_y.float()
        mask_x_float = mask_x.float()


        torch.manual_seed(mask_x.sum()+mask_y.sum())

        y_inds = torch.multinomial(mask_y_float, 1)
        x_inds = torch.multinomial(mask_x_float, 1)

        selected_x = torch.zeros_like(mask_x)
        # pdb.set_trace()
        selected_x[(torch.arange(0, mask_x.shape[0]).to(x_inds.dtype), x_inds[:,0])] = True
        inds_x = torch.where(mask_x * ~selected_x)

        indices = torch.randperm(len(inds_x[0]))[:int(np.floor(len(inds_x[0])*sparsity))]
        select_indices = (inds_x[0][indices], inds_x[1][indices])
        selected_x[select_indices] = True

        mask_x = selected_x

        selected_y = torch.zeros_like(mask_y)
        if mask_y.shape[0] > 0:
            selected_y[(torch.arange(0, mask_y.shape[0]).to(y_inds.dtype), y_inds[:,0])] = True
        # selected_y[:, y_inds] = True
            inds_y = torch.where(mask_y * ~selected_y)
            indices = torch.randperm(len(inds_y[0]))[:int(np.floor(len(inds_y[0])*sparsity))]
            # if len(indices) == 0:
            select_indices_y = (inds_y[0][indices], inds_y[1][indices])
            selected_y[select_indices_y] = True

            mask_y = selected_y
        
        # nan to zeros
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)


        x_vals.append(x[sorted_idx])
        x_time.append(t[sorted_idx])
        x_mask.append(mask_x[sorted_idx])

        y_time.append(t_target)
        y_vals.append(y)
        y_mask.append(mask_y)
        
        context_x.append(torch.cat([t, t_target], dim = 0))
        x_vals_temp = torch.zeros_like(x)
        y_vals_temp = torch.zeros_like(y)
        context_vals.append(torch.cat([x, y_vals_temp], dim=0))
        context_mask.append(torch.cat([mask_x, y_vals_temp], dim=0))
        # context_y = torch.cat([context_vals, context_mask], dim=2)

        target_vals.append(torch.cat([x_vals_temp, y], dim=0))
        target_mask.append(torch.cat([x_vals_temp, mask_y], dim=0))
        # target_y = torch.cat([target_vals, target_mask], dim=2)

    return Batch(
        x_time=pad_sequence(context_x, batch_first=True).squeeze(),
        x_vals=pad_sequence(context_vals, batch_first=True, padding_value=0).squeeze(),
        x_mask=pad_sequence(context_mask, batch_first=True).squeeze(),
        y_time=pad_sequence(context_x, batch_first=True).squeeze(),
        y_vals=pad_sequence(target_vals, batch_first=True, padding_value=0).squeeze(),
        y_mask=pad_sequence(target_mask, batch_first=True).squeeze(),
    )

dloader_config_train = {
    "batch_size": ARGS.batch_size,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
    "num_workers": 4,
    "collate_fn": lambda b: tsdm_sparse_collate(b, ARGS.sparsity),
}
