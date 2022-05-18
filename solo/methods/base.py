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

from argparse import ArgumentParser
from ctypes.wintypes import tagRECT
from functools import partial
from turtle import clear
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
from solo.utils import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from solo.methods.resnet18k import make_resnet18k
# import solo.methods.resnet.resnet as resnet_mine
import solo.methods.ResCg.rescg as resnet18_cg
from solo.losses.regularization_channel import* 

import sys

from solo.utils.backbones import (
    poolformer_m36,
    poolformer_m48,
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)
from solo.utils.knn import WeightedKNNClassifier
from solo.utils.lars import LARSWrapper
from solo.utils.metrics import accuracy_at_k, weighted_mean
from solo.utils.momentum import MomentumUpdater, initialize_momentum_params
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torchvision.models import  resnet18, resnet50 


def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs

def getAvgMeter(num):
    return [metrics.AverageMeter() for _ in range(num)]

class ExpAnnealing(object):
    r"""
    Args:
        T_max (int): Maximum number of iterations.
        eta_ini (float): Initial density. Default: 1.
        eta_min (float): Minimum density. Default: 0.
    """

    def __init__(self, T_ini, eta_ini=1, eta_final=0, up=False, alpha=1):
        self.T_ini = T_ini
        self.eta_final = eta_final
        self.eta_ini = eta_ini
        self.up = up
        self.last_epoch = 0
        self.alpha = alpha

    def get_lr(self, epoch):
        if epoch < self.T_ini:
            return self.eta_ini
        elif self.up:
            return self.eta_ini + (self.eta_final-self.eta_ini) * (1-
                   math.exp(-self.alpha*(epoch-self.T_ini)))
        else:
            return self.eta_final + (self.eta_ini-self.eta_final) * math.exp(
                   -self.alpha*(epoch-self.T_ini))

    def step(self):
        self.last_epoch += 1
        return self.get_lr(self.last_epoch)

class BaseMethod(pl.LightningModule):

    _SUPPORTED_BACKBONES = {
        "resnet18": resnet18,
        # "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
        "poolformer_s12": poolformer_s12,
        "poolformer_s24": poolformer_s24,
        "poolformer_s36": poolformer_s36,
        "poolformer_m36": poolformer_m36,
        "poolformer_m48": poolformer_m48,
    }

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        backbone_args: dict,
        max_epochs: int,
        batch_size: int,
        optimizer: str,
        lars: bool,
        lr: float,
        weight_decay: float,
        classifier_lr: float,
        exclude_bias_n_norm: bool,
        accumulate_grad_batches: Union[int, None],
        extra_optimizer_args: Dict,
        scheduler: str,
        den_target:float,
        gamma:float,
        lbda: float,
        alpha: float,
        min_lr: float,
        warmup_start_lr: float,
        warmup_epochs: float,
        num_large_crops: int,
        num_small_crops: int,
        eta_lars: float = 1e-3,
        grad_clip_lars: bool = False,
        lr_decay_steps: Sequence = None,
        knn_eval: bool = False,
        knn_k: int = 20,
        # width: int = 64,
        **kwargs,
    ):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        Args:
            backbone (str): architecture of the base backbone.
            num_classes (int): number of classes.
            backbone_params (dict): dict containing extra backbone args, namely:
                #! optional, if it's not present, it is considered as False
                cifar (bool): flag indicating if cifar is being used.
                #! only for resnet
                zero_init_residual (bool): change the initialization of the resnet backbone.
                #! only for vit
                patch_size (int): size of the patches for ViT.
            max_epochs (int): number of training epochs.
            batch_size (int): number of samples in the batch.
            optimizer (str): name of the optimizer.
            lars (bool): flag indicating if lars should be used.
            lr (float): learning rate.
            weight_decay (float): weight decay for optimizer.
            classifier_lr (float): learning rate for the online linear classifier.
            exclude_bias_n_norm (bool): flag indicating if bias and norms should be excluded from
                lars.
            accumulate_grad_batches (Union[int, None]): number of batches for gradient accumulation.
            extra_optimizer_args (Dict): extra named arguments for the optimizer.
            scheduler (str): name of the scheduler.
            min_lr (float): minimum learning rate for warmup scheduler.
            warmup_start_lr (float): initial learning rate for warmup scheduler.
            warmup_epochs (float): number of warmup epochs.
            num_large_crops (int): number of big crops.
            num_small_crops (int): number of small crops .
            eta_lars (float): eta parameter for lars.
            grad_clip_lars (bool): whether to clip the gradients in lars.
            lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                step. Defaults to None.
            knn_eval (bool): enables online knn evaluation while training.
            knn_k (int): the number of neighbors to use for knn.

        .. note::
            When using distributed data parallel, the batch size and the number of workers are
            specified on a per process basis. Therefore, the total batch size (number of workers)
            is calculated as the product of the number of GPUs with the batch size (number of
            workers).

        .. note::
            The learning rate (base, min and warmup) is automatically scaled linearly based on the
            batch size and gradient accumulation.

        .. note::
            For CIFAR10/100, the first convolutional and maxpooling layers of the ResNet backbone
            are slightly adjusted to handle lower resolution images (32x32 instead of 224x224).

        """
        # print("-----",kwargs["width"])
        super().__init__()
        
        # resnet backbone related
        self.backbone_args = backbone_args
        
        # training related
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lars = lars
        self.lr = lr
        self.weight_decay = weight_decay
        self.classifier_lr = classifier_lr
        self.exclude_bias_n_norm = exclude_bias_n_norm
        self.accumulate_grad_batches = accumulate_grad_batches
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.lr_decay_steps = lr_decay_steps
        self.min_lr = min_lr
        
        self.den_target = den_target
        self.lbda = lbda
        self.gamma = gamma
        self.alpha = alpha
       
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.num_large_crops = num_large_crops
        self.num_small_crops = num_small_crops
        self.eta_lars = eta_lars
        self.grad_clip_lars = grad_clip_lars
        self.knn_eval = knn_eval
        self.knn_k = knn_k

        self.width = kwargs["width"]
       
        #add dataset key
        self.dataset = kwargs["dataset"]
       
        
        # multicrop
        self.num_crops = self.num_large_crops + self.num_small_crops

        # all the other parameters
        self.extra_args = kwargs

        # turn on multicrop if there are small crops
        self.multicrop = self.num_small_crops != 0

        # if accumulating gradient then scale lr
        if self.accumulate_grad_batches:
            self.lr = self.lr * self.accumulate_grad_batches
            self.classifier_lr = self.classifier_lr * self.accumulate_grad_batches
            self.min_lr = self.min_lr * self.accumulate_grad_batches
            self.warmup_start_lr = self.warmup_start_lr * self.accumulate_grad_batches

        assert backbone in BaseMethod._SUPPORTED_BACKBONES
        self.base_model = self._SUPPORTED_BACKBONES[backbone]

        self.backbone_name = backbone #for the moment resnet18 supported gating

        # initialize backbone
        kwargs = self.backbone_args.copy()
        # cifar = kwargs.pop("cifar", False)
        cifar = kwargs["cifar"]
       
        
        # swin specific
        if "swin" in self.backbone_name and cifar:
            kwargs["window_size"] = 4
        
        #NOTE only for Cifar
        # self.backbone = make_resnet18k(k=self.width, **kwargs)
        # self.features_dim = self.backbone.num_features 

        #NOTE Removed this part
        # print(kwargs)
        
        # self.backbone = self.base_model(**kwargs)
        # kwargs["width"] = self.width #NOTE no need
        
        self.backbone = resnet18_cg.resdg18(**kwargs)
        self.features_dim = self.backbone.inplanes
        
        #set criterion
        self.backbone.set_criterion(Loss())
        print("Density_Target; {}, Gamma: {}, Lambda: {}, Alpha: {}".format(self.den_target,
                                                                            self.gamma,
                                                                            self.lbda,
                                                                            self.alpha))
        self.p_anneal = ExpAnnealing(0, 1, 0, alpha=self.alpha)
        self.p = self.p_anneal.get_lr(self.current_epoch)

        
        
        # if "resnet" in self.backbone_name:
        #     self.features_dim = self.backbone.inplanes #TODO for further feature dim
        #     # remove fc layer
        #     self.backbone.fc = nn.Identity()
        #     if cifar:
        #         self.backbone.conv1 = nn.Conv2d(
        #             3, self.width, kernel_size=3, stride=1, padding=2, bias=False
        #         )
        #         self.backbone.maxpool = nn.Identity()
        # else:
        #     self.features_dim = self.backbone.num_features
    
       
        #NOTE Removed this for SSL Training
        # self.classifier = nn.Linear(self.features_dim, num_classes)

        if self.knn_eval:
            self.knn = WeightedKNNClassifier(k=self.knn_k, distance_fx="euclidean")

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds shared basic arguments that are shared for all methods.

        Args:
            parent_parser (ArgumentParser): argument parser that is used to create a
                argument group.

        Returns:
            ArgumentParser: same as the argument, used to avoid errors.
        """

        parser = parent_parser.add_argument_group("base")

        # backbone args
        SUPPORTED_BACKBONES = BaseMethod._SUPPORTED_BACKBONES

        parser.add_argument("--backbone", choices=SUPPORTED_BACKBONES, type=str)
        parser.add_argument("--width", type=int, help="resnet width", default=64)
        # extra args for resnet
        parser.add_argument("--zero_init_residual", action="store_true")
        # extra args for ViT
        parser.add_argument("--patch_size", type=int, default=16)

        # general train
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=0.3)
        parser.add_argument("--classifier_lr", type=float, default=0.3)
        parser.add_argument("--weight_decay", type=float, default=0.0001)
        parser.add_argument("--num_workers", type=int, default=4)


        #add gating related args
        parser.add_argument('--den-target', default=0.5, type=float, help='target density of the mask.')
        parser.add_argument('--lbda', default=5, type=float, help='penalty factor of the L2 loss for mask.')
        parser.add_argument('--gamma', default=1, type=float, help='penalty factor of the L2 loss for balance gate.')
        parser.add_argument('--alpha', default=5e-2, type=float, help='alpha in exp annealing.')

        # wandb
        parser.add_argument("--name")
        parser.add_argument("--project")
        parser.add_argument("--entity", default=None, type=str)
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--offline", action="store_true")

        # optimizer
        SUPPORTED_OPTIMIZERS = ["sgd", "adam", "adamw"]

        parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
        parser.add_argument("--lars", action="store_true")
        parser.add_argument("--grad_clip_lars", action="store_true")
        parser.add_argument("--eta_lars", default=1e-3, type=float)
        parser.add_argument("--exclude_bias_n_norm", action="store_true")

        # scheduler
        SUPPORTED_SCHEDULERS = [
            "reduce",
            "cosine",
            "warmup_cosine",
            "step",
            "exponential",
            "none",
        ]

        parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce")
        parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
        parser.add_argument("--min_lr", default=0.0, type=float)
        parser.add_argument("--warmup_start_lr", default=0.003, type=float)
        parser.add_argument("--warmup_epochs", default=10, type=int)

        # DALI only
        # uses sample indexes as labels and then gets the labels from a lookup table
        # this may use more CPU memory, so just use when needed.
        parser.add_argument("--encode_indexes_into_labels", action="store_true")

        # online knn eval
        parser.add_argument("--knn_eval", action="store_true")
        parser.add_argument("--knn_k", default=20, type=int)

        return parent_parser

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """
        # print("I'am in here")
        param_dict = dict(self.backbone.named_parameters())
        params = []
        # BN_name_pool = []
        # for m_name, m in model.named_modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         BN_name_pool.append(m_name + '.weight')
        #         BN_name_pool.append(m_name + '.bias')
        for key, value in param_dict.items():
            if 'mask' in key:
                params += [{'params': [value], 'lr': self.lr, 'weight_decay': 0.}]
            else:
                params += [{'params':[value]}]
        return params
        # return [
        #     {"name": "backbone", "params": params},
        #     # {
        #     #     "name": "classifier",
        #     #     "params": self.classifier.parameters(),
        #     #     "lr": self.classifier_lr,
        #     #     "weight_decay": 0,
        #     # },
        # ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam, adamw)")

        # create optimizer
        optimizer = optimizer(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )
        # optionally wrap with lars
        if self.lars:
            assert self.optimizer == "sgd", "LARS is only compatible with SGD."
            optimizer = LARSWrapper(
                optimizer,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            )

        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.max_epochs,
                warmup_start_lr=self.warmup_start_lr,
                eta_min=self.min_lr,
            )
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, self.max_epochs, eta_min=self.min_lr)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]

    def forward(self, *args, **kwargs) -> Dict:
        """Dummy forward, calls base forward."""

        return self.base_forward(*args, **kwargs)

    def base_forward(self, X: torch.Tensor) -> Dict:
        """Basic forward that allows children classes to override forward().

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """
        # create input for Rescg:  forward(x, den_target, lbda, gamma, p) 
        out = self.backbone(X, self.den_target, self.lbda, self.gamma, self.p)

        #No logits for SSL
        # logits = self.classifier(feats.detach())
        return {**out}

    def _base_shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """
        
        out = self.base_forward(X)
        #NOTE Keep only feats

        # logits = out["logits"]

        # loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # # handle when the number of classes is smaller than 5
        # top_k_max = min(5, logits.size(1))
        # acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

        # return {**out, "loss": loss, "acc1": acc1, "acc5": acc5}
        # print(out["feats"].shape)
        return {**out}
        
    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits.
        """

        _, X, targets = batch

        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.num_crops
        batch_size = X[0].size(0)
        # print(batch_size)
        outs = [self._base_shared_step(x, targets) for x in X[: self.num_large_crops]]
        # print(len(outs), outs[0].keys())
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}
        # print(outs.keys(), len(outs["feats"]))
        # ----------------------------------------------------------------
        flops_real = outs["flops_real"][0]  
        flops_mask = outs["flops_mask"][0] 
        flops_ori = outs["flops_ori"][0]  

        flops_tensor, flops_conv1, flops_fc = flops_real[0], flops_real[1], flops_real[2]
        # block flops
        flops_conv = flops_tensor[0:batch_size,:].mean(0).sum()
        flops_mask = flops_mask.mean(0).sum()
        flops_ori = flops_ori.mean(0).sum() + flops_conv1.mean() + flops_fc.mean()
        flops_real = flops_conv + flops_mask + flops_conv1.mean() + flops_fc.mean()
        # print(flops_ori, flops_real, flops_real/flops_ori)
        log = {}
        log = {"train_flops_real_ratio_orig": flops_real/flops_ori,"train_flops_real": flops_real, "train_flops_orig": flops_ori}
        self.log_dict(log, on_step = True, on_epoch=False, sync_dist=True)
        # -----------------------------------------------------------------

        if self.multicrop:
            outs["feats"].extend([self.backbone(x) for x in X[self.num_large_crops :]])
        
        # flops_real = [sum(value)/2 for value in zip(outs["flops_real"][0],outs["flops_real"][1])]
        # flops_mask = [sum(value)/2 for value in zip(outs["flops_mask"][0],outs["flops_mask"][1])]
        # flops_ori = [sum(value)/2 for value in zip(outs["flops_ori"][0], outs["flops_ori"][1])]

        # flops_conv, _, _, _, _ = metrics.analyse_flops(flops_real, flops_mask, flops_ori, batch_size)
        
        #NOTE comment this not required
        """
            # # loss and stats
            # outs["loss"] = sum(outs["loss"]) / self.num_large_crops
            # outs["acc1"] = sum(outs["acc1"]) / self.num_large_crops
            # outs["acc5"] = sum(outs["acc5"]) / self.num_large_crops

            # metrics = {
            #     "train_class_loss": outs["loss"],
            #     "train_acc1": outs["acc1"],
            #     "train_acc5": outs["acc5"],
            # }

            # self.log_dict(metrics, on_epoch=True, sync_dist=True)

           
        """
        if self.knn_eval:
            targets = targets.repeat(self.num_large_crops)
            mask = targets != -1
            self.knn(
                train_features=torch.cat(outs["feats"][: self.num_large_crops])[mask].detach(),
                train_targets=targets[mask],
            )

        return outs

    def on_train_epoch_start(self) -> None:
        # self.start_epoch = time.time()
        
        self.p = self.p_anneal.get_lr(self.current_epoch)
        self.train_block_flop= getAvgMeter(1)



    # #Override validation_step and validation_epoch_end 
    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """

        X, targets = batch
        batch_size = targets.size(0)

        out = self._base_shared_step(X, targets)

        if self.knn_eval and not self.trainer.sanity_checking:
            self.knn(test_features=out.pop("feats").detach(), test_targets=targets.detach())

        flops_real = out["flops_real"]
        flops_mask = out["flops_mask"]
        flops_ori  = out["flops_ori"]

        self.val_rlosses.update(out["rloss"].mean().item(), batch_size)
        self.val_blosses.update(out["bloss"].mean().item(), batch_size)

        flops_conv, self.flops_mask, self.flops_ori, self.flops_conv1, self.flops_fc = metrics.analyse_flops(
                                              flops_real, flops_mask, flops_ori, batch_size)
        self.val_block_flop.update(flops_conv, batch_size)
        log = {}

        log = {"val_sparsity_loss": self.val_rlosses.avg, "val_bound_loss": self.val_blosses.avg}

        self.log_dict(log, on_step = True, on_epoch=False, sync_dist=True)
        # metrics = {
        #     "batch_size": batch_size,
        #     "val_loss": out["loss"],
        #     "val_acc1": out["acc1"],
        #     "val_acc5": out["acc5"],
        # }
        return out

    def on_validation_epoch_start(self) -> None:
        self.val_rlosses, self.val_blosses,  self.val_block_flop= getAvgMeter(3)

    
    
    
    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        #NOTE No Loss
        """
            # val_loss = weighted_mean(outs, "val_loss", "batch_size")
            # val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
            # val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")

            # log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        """
        flops = (self.val_block_flop.avg[-1]+self.flops_mask[-1]+self.flops_conv1.mean()+self.flops_fc.mean())/1024
        flops_per = (self.val_block_flop.avg[-1]+self.flops_mask[-1]+self.flops_conv1.mean()+self.flops_fc.mean())/(
                         self.flops_ori[-1]+self.flops_conv1.mean()+self.flops_fc.mean())*100
        log ={}
        if self.knn_eval and not self.trainer.sanity_checking:
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log = {
                   "val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5,
                   "val_flops": flops, "val_flops_percent": flops_per,
                   "val_block_flops": self.val_block_flop.avg[-1]
                   }

        self.log_dict(log, sync_dist=True, on_step=False, on_epoch=True)

