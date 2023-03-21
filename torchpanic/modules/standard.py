# This file is part of Prototypical Additive Neural Network for Interpretable Classification (PANIC).
#
# PANIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PANIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PANIC. If not, see <https://www.gnu.org/licenses/>.
from functools import partial
from typing import Any, List

import torch
from torch import nn

from .base import BaseModule


class StandardModule(BaseModule):
    def __init__(
        self,
        net: nn.Module,
        lr: float = 0.001,
        num_classes: int = 3,
        weight_decay: float = 0.0005,
        output_penalty_weight: float = 0.001,
        **kwargs,
    ):
        super().__init__(
            net=net,
            num_classes=num_classes,
        )

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        # loss function
        if num_classes > 2:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()

    def step(self, batch: Any):
        x, _, _, y = batch
        logits, terms = self.forward(x)
        if self.hparams.num_classes < 3:
            logits = logits.squeeze()
            probs = torch.sigmoid(logits)
            preds = torch.argmax(torch.stack((1.0 - probs, probs), dim=-1), dim=-1)
        else:
            preds = torch.argmax(logits, dim=1)

        loss = self.criterion(logits, y)
        if self.hparams.output_penalty_weight > 0:
            output_norm = torch.linalg.norm(terms, dim=[1, 2])
            loss = loss + self.hparams.output_penalty_weight * output_norm.mean()
        return loss, preds, y.long()

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self._log_train_metrics(loss, preds, targets)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self._update_validation_metrics(loss, preds, targets)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        self._log_validation_metrics()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        self._update_test_metrics(loss, preds, targets)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        self._log_test_metrics()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.AdamW(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        n_iters = len(self.trainer.datamodule.train_dataloader())
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.hparams.lr / 10,
            max_lr=self.hparams.lr,
            step_size_up=5 * n_iters,
            step_size_down=5 * n_iters,
            cycle_momentum=False,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
