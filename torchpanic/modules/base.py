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
import logging
from typing import Any, List

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torchmetrics import Accuracy, ConfusionMatrix, MaxMetric

from .utils import get_git_hash

from ..datamodule.modalities import DataPointType


LOG = logging.getLogger(__name__)


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.net = net
        task = "binary" if num_classes <= 2 else "multiclass"
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_acc = Accuracy(task=task, num_classes=num_classes)
        self.test_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_cmat = ConfusionMatrix(task=task, num_classes=num_classes)
        self.test_cmat = ConfusionMatrix(task=task, num_classes=num_classes)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_bacc_best = MaxMetric()

    def _get_balanced_accuracy_from_confusion_matrix(self, confusion_matrix: ConfusionMatrix):
        # Confusion matrix whose i-th row and j-th column entry indicates
        # the number of samples with true label being i-th class and
        # predicted label being j-th class.
        cmat = confusion_matrix.compute()
        per_class = cmat.diag() / cmat.sum(dim=1)
        per_class = per_class[~torch.isnan(per_class)]  # remove classes that are not present in this dataset
        LOG.debug("Confusion matrix:\n%s", cmat)

        return per_class.mean()

    def _log_train_metrics(
        self, loss: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor,
    ) -> None:
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        acc = self.train_acc(preds, targets)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def _update_validation_metrics(
        self, loss: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor,
    ) -> None:
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.val_acc.update(preds, targets)
        self.val_cmat.update(preds, targets)

    def _log_validation_metrics(self) -> None:
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.log("val/acc", acc, on_epoch=True)

        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True)

        # compute balanced accuracy
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.val_cmat)
        self.val_bacc_best.update(bacc)
        self.log("val/bacc", bacc, on_epoch=True, prog_bar=True)
        self.log("val/bacc_best", self.val_bacc_best.compute(), on_epoch=True)

        # reset metrics at the end of every epoch
        self.val_acc.reset()
        self.val_cmat.reset()

    def _update_test_metrics(
        self, loss: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor,
    ) -> None:
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.test_acc.update(preds, targets)
        self.test_cmat.update(preds, targets)

    def _log_test_metrics(self) -> None:
        acc = self.test_acc.compute()
        self.log("test/acc", acc)

        # compute balanced accuracy
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.test_cmat)
        self.log("test/bacc", bacc)

        # reset metrics at the end of every epoch
        self.test_acc.reset()
        self.test_cmat.reset()

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        self.val_bacc_best.reset()

        if isinstance(self.logger, TensorBoardLogger):
            tb_logger = self.logger.experiment
            # tb_logger.add_hparams({"git-commit": get_git_hash()}, {"hp_metric": -1})
            tb_logger.flush()

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # reset metrics at the end of every epoch
        self.train_acc.reset()

    def forward(self, x: DataPointType):
        return self.net(x)
