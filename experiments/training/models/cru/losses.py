# Modeling Irregular Time Series with Continuous Recurrent Units (CRUs)
# Copyright (c) 2022 Robert Bosch GmbH
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This source code is derived from Pytorch RKN Implementation (https://github.com/ALRhub/rkn_share)
# Copyright (c) 2021 Philipp Becker (Autonomous Learning Robots Lab @ KIT)
# licensed under MIT License
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import numpy as np
import torch


# taken from https://github.com/ALRhub/rkn_share/ and modified
def rmse(target, predicted, mask=None):
    """Root Mean Squared Error"""
    if mask is None:
        mask = torch.ones_like(predicted)
    else:
        assert mask.shape == predicted.shape == target.shape
    return torch.sqrt(
        torch.sum(mask * torch.square(target - predicted)) / torch.sum(mask)
    )


# taken from https://github.com/ALRhub/rkn_share/ and modified
def mse(target, predicted, mask=None):
    """Mean Squared Error"""
    if mask is None:
        mask = torch.ones_like(predicted)
    else:
        assert mask.shape == predicted.shape == target.shape
    return torch.sum(mask * torch.square(target - predicted)) / torch.sum(mask)


def mae(target, predicted, mask=None):
    """Mean Squared Error"""
    if mask is None:
        mask = torch.ones_like(predicted)
    else:
        assert mask.shape == predicted.shape == target.shape
    return torch.sum(mask * torch.abs(target - predicted)) / torch.sum(mask)


# taken from https://github.com/ALRhub/rkn_share/ and modified
def bernoulli_nll(targets, predictions, uint8_targets=False):
    """Computes Binary Cross Entropy
    :param targets: target sequence
    :param predictions: predicted sequence
    :param uint8_targets: if true it is assumed that the targets are given in uint8 (i.e. the values are integers
    between 0 and 255), thus they are devided by 255 to get "float image representation"
    :return: Binary Crossentropy between targets and prediction
    """
    if uint8_targets:
        targets = targets / 255.0
    point_wise_error = -1 * (
        targets * torch.log(predictions + 1e-12)
        + (1 - targets) * torch.log(1 - predictions + 1e-12)
    )
    red_axis = [i + 2 for i in range(len(targets.shape) - 2)]
    sample_wise_error = torch.sum(point_wise_error, axis=red_axis)
    return torch.mean(sample_wise_error)


# new code component
def GaussianNegLogLik(
    targets, pred_mean, pred_variance, mask=None, normalize_dim=False
):
    """Computes Gaussian Negaitve Loglikelihood
    :param targets: target sequence
    :param pred_mean: output sequence
    :param pred_var: output variance
    :param mask: target mask
    :param normalize_dim: if to normalize over the number of observed dimensions
    :return: Gaussian Negative Loglikelihood of target sequence
    """
    assert (
        pred_mean.shape == targets.shape == pred_variance.shape
    ), f"pred_mean {pred_mean.shape} targets {targets.shape} pred_variance {pred_variance.shape}"

    epsilon = 1e-6 * torch.ones_like(pred_mean)
    pred_variance = torch.maximum(pred_variance, epsilon)

    if mask == None:
        mask = torch.ones_like(pred_mean)

    # sum over dimensions
    const = np.log(2 * np.pi)
    sample_dim_time_wise = mask * (
        torch.log(pred_variance)
        + torch.square(pred_mean - targets) / pred_variance
        + const
    )
    sample_time_wise = 0.5 * torch.sum(sample_dim_time_wise, -1)

    # divide by number of observed dimensions if normalize_dim
    if normalize_dim:
        num_dim_observed = torch.sum(mask, -1)
        sample_time_wise = sample_time_wise / num_dim_observed

    # mean over time steps
    sample_wise = torch.mean(sample_time_wise, 1)

    return torch.mean(sample_wise)
