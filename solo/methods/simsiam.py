# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from typing import Any, Dict, List, Sequence
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simsiam import simsiam_loss_func
from solo.methods.base import BaseMethod


class SimSiam(BaseMethod):
    def __init__(
        self,
        base_proj_output_dim: int,
        base_proj_hidden_dim: int,
        base_pred_hidden_dim: int,
        **kwargs,
    ):
        """Implements SimSiam (https://arxiv.org/abs/2011.10566).

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            pred_hidden_dim (int): number of neurons of the hidden layers of the predictor.
        """
        # print(kwargs)
        super().__init__(**kwargs)
        
        
        # projector
        proj_hidden_dim = self.width*base_proj_hidden_dim
        proj_output_dim = self.width*base_proj_output_dim

        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim, affine=False),
        )
        self.projector[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # predictor
        pred_hidden_dim = self.width*base_pred_hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim, bias=False),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimSiam, SimSiam).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simsiam")

        # # projector
        # parser.add_argument("--proj_output_dim", type=int, default=128)
        # parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # # predictor
        # parser.add_argument("--pred_hidden_dim", type=int, default=512)

        #NOTE Consider them as base prediction and prediction dimensions and width to get actual dimension 
        # accordingly. for k=64 proj_dim(hid and out =2048) and pred_hidden/out(512/2048)  
        # projector
        parser.add_argument("--base_proj_output_dim", type=int, default=32)
        parser.add_argument("--base_proj_hidden_dim", type=int, default=32)

        # predictor
        parser.add_argument("--base_pred_hidden_dim", type=int, default=8)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params: List[dict] = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters(), "static_lr": True},
        ]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the backbone, the projector and the predictor.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected and predicted features.
        """
        
        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimSiam reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images
            batch_idx (int): index of the batch

        Returns:
            torch.Tensor: total loss composed of SimSiam loss and classification loss
        """

        out = super().training_step(batch, batch_idx)
        

        # class_loss = out["loss"] #NOTE No class loss
        feats1, feats2 = out["feats"]

        z1 = self.projector(feats1)
        z2 = self.projector(feats2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # ------- contrastive loss -------
        #NOTE function simsiam_loss includes .detach()
        neg_cos_sim = simsiam_loss_func(p1, z2)/ 2 + simsiam_loss_func(p2, z1) / 2
        # -------------------------------------------------------
        sp1_l = out["rloss"][0].mean() + out["bloss"][0].mean()
        sp2_l = out["rloss"][1].mean() + out["bloss"][1].mean()
        #------------------------------------------------------
        # calculate std of features
        z1_std = F.normalize(z1, dim=-1).std(dim=0).mean()
        z2_std = F.normalize(z2, dim=-1).std(dim=0).mean()
        z_std = (z1_std + z2_std) / 2

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
            "train_total_sparsity": sp1_l + sp2_l,
            "train_spar_loss": out["rloss"][0].mean() + out["rloss"][1].mean(),
            "train_bound_loss": out["bloss"][0].mean() + out["bloss"][1].mean()
           
        }
        self.log_dict(metrics, on_step = True, on_epoch=True, sync_dist=True)
        return neg_cos_sim + sp1_l + sp2_l
        # return neg_cos_sim + class_loss
    
    


    # def on_train_epoch_end(self) -> None:
    #     #TODO write training eval script for K-NN
    #     print("time for each epoch {}".format(time.time() - self.start_epoch))

    