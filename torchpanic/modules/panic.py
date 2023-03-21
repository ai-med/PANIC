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
import math
from pathlib import Path
from typing import Any, List
import warnings

import numpy as np
# from skimage.transform import resize

from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.nn.functional import l1_loss
from torchmetrics import MaxMetric

from ..datamodule.modalities import BatchWithLabelType, ModalityType
from .base import BaseModule

LOG = logging.getLogger(__name__)

STAGE2FLOAT = {"warmup": 0.0, "warmup_protonet": 1.0, "all": 2.0, "nam_only": 3.0}


# @torch.no_grad()
# def init_weights(m: torch.Tensor):
#     if isinstance(m, nn.Conv3d):
#         # every init technique has an underscore _ in the name
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if getattr(m, "bias", None) is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.BatchNorm3d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)


class PANIC(BaseModule):
    def __init__(
        self,
        net: torch.nn.Module,
        weight_decay_nam: float = 0.0,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        l_clst: float = 0.8,
        l_sep: float = 0.08,
        l_occ: float = 1e-4,
        l_affine: float = 1e-4,
        l_nam: float = 1e-4,
        epochs_all: int = 3,
        epochs_nam: int = 4,
        epochs_warmup: int = 10,
        enable_checkpointing: bool = True,
        monitor_prototypes: bool = False,  # wether to save prototypes of all push epochs or just the best one
        enable_save_embeddings: bool = False,
        enable_log_prototypes: bool = False,
        **kwargs,
    ):
        super().__init__(
            net=net,
            num_classes=net.num_classes,
        )
        self.automatic_optimization = False

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])
        self._current_stage = None
        self._current_optimizer = None

        # FIXME only picks PET
        self.image_modality = ModalityType.PET

        self.net = net
        if self.net.two_d:
            self.reduce_dims = (2, 3,)  # inputs are 4d tensors
        else:
            self.reduce_dims = (2, 3, 4,)  # inputs are 5d tensors

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        self.val_bacc_save = MaxMetric()

    def _cluster_losses(self, similarities, y):
        prototypes_of_correct_class = torch.t(self.net.prototype_class_identity[:, y]).bool()
        # select mask for each sample in batch. Shape is (bs, n_prototypes)
        other_tensor = torch.tensor(0, device=similarities.device)  # minimum value of the cosine similarity
        similarities_correct_class = similarities.where(prototypes_of_correct_class, other=other_tensor)
        # min value of cos similarity doesnt effect result
        similarities_incorrect_class = similarities.where(~prototypes_of_correct_class, other=other_tensor)
        # same here: if distance to other protos of other class is the minimum value
        # of the cos similarity, they are distant to eachother!
        clst = 1 - torch.max(similarities_correct_class, dim=1).values.mean()
        sep = torch.max(similarities_incorrect_class, dim=1).values.mean()

        return clst, sep

    def _occurence_map_losses(self, occurrences, x_raw, aug):
        with torch.no_grad():
            _, _, occurrences_raw = self.net.forward_image(x_raw)

            if self.net.two_d:
                occurrences_raw = occurrences_raw.unsqueeze(-1)
            occurrences_raw = occurrences_raw.cpu()

            for i, aug_i in enumerate(aug):
                occurrences_raw[i] = aug_i(occurrences_raw[i])

            if self.net.two_d:
                occurrences_raw = occurrences_raw.squeeze(-1)
            occurrences_raw = occurrences_raw.to(occurrences.device)

        affine = l1_loss(occurrences_raw, occurrences)

        # l1 penalty on occurence maps
        l1 = torch.linalg.vector_norm(occurrences, ord=1, dim=self.reduce_dims).mean()
        l1_norm = math.prod(occurrences.size()[1:])  # omit batch dimension
        l1 = (l1 / l1_norm).mean()
        return affine, l1

    def _classification_loss(self, logits, targets):
        preds = torch.argmax(logits, dim=1)

        xentropy = self.criterion(logits, targets)
        return xentropy, preds

    def forward(self, batch: BatchWithLabelType):
        x = batch[0]
        image = x[self.image_modality]
        tabular = x[ModalityType.TABULAR]

        return self.net(image, tabular)

    def on_train_epoch_start(self) -> None:
        cur_stage = self._get_current_stage()
        LOG.info("Epoch %d, optimizing %s", self.trainer.current_epoch, cur_stage)

        self.log("train/stage", STAGE2FLOAT[cur_stage])

        optim_warmup, optim_warmup_protonet, optim_all, optim_nam = self.optimizers()
        scheduler_warmup, scheduler_all, scheduler_nam = self.lr_schedulers()
        if cur_stage == "warmup":
            opt = optim_warmup
            sched = scheduler_warmup
        elif cur_stage == "warmup_protonet":
            opt = optim_warmup_protonet
            sched = scheduler_warmup
        elif cur_stage == "all":
            opt = optim_all
            sched = scheduler_all
        elif cur_stage == "nam_only":
            opt = optim_nam
            sched = scheduler_nam
            self.push_prototypes()
        else:
            raise AssertionError()

        self._current_stage = cur_stage
        self._current_optimizer = opt
        self._current_scheduler = sched

    def training_step(self, batch: BatchWithLabelType, batch_idx: int):
        cur_stage = self._current_stage

        x_raw, aug, y = batch[1:]
        x_raw = x_raw[self.image_modality]
        aug = aug[self.image_modality]

        logits, similarities, occurrences, nam_terms = self.forward(batch)
        xentropy, preds = self._classification_loss(logits, y)
        self.log("train/xentropy", xentropy)
        losses = [xentropy]

        if cur_stage != "nam_only":
            # cluster and seperation cost
            clst, sep = self._cluster_losses(similarities, y)
            losses.append(self.hparams.l_clst * clst)
            losses.append(self.hparams.l_sep * sep)

            self.log("train/clst", clst)
            self.log("train/sep", sep)

            # regularization of occurrence map
            affine, l1 = self._occurence_map_losses(occurrences, x_raw, aug)
            losses.append(self.hparams.l_affine * affine)
            losses.append(self.hparams.l_occ * l1)

            self.log("train/affine", affine)
            self.log("train/l1", l1)

        if cur_stage != "occ_and_feats":
            # l2 penalty on terms of nam
            start_index = self.net.n_prototypes_per_class
            nam_penalty = nam_terms[:, start_index:, :].square().sum(dim=1).mean()
            losses.append(self.hparams.l_nam * nam_penalty)

            self.log("train/nam_l2", nam_penalty)

        loss = sum(losses)

        opt = self._current_optimizer
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self._current_scheduler.step()

        self._log_train_metrics(loss, preds, y)

        return {"loss": loss, "preds": preds, "targets": y}

    def _log_train_metrics(
        self, loss: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor,
    ) -> None:
        self.log(f"train/loss/{self._current_stage}", loss, on_step=False, on_epoch=True, prog_bar=False)
        acc = self.train_acc(preds, targets)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def _get_current_stage(self, epoch=None):
        total = self.hparams.epochs_all + self.hparams.epochs_nam

        stage = "nam_only"
        if self.current_epoch < 0.5 * self.hparams.epochs_warmup:
            stage = "warmup"
        elif self.current_epoch < self.hparams.epochs_warmup:
            stage = "warmup_protonet"
        elif (self.current_epoch - self.hparams.epochs_warmup) % total < self.hparams.epochs_all:
            stage = "all"

        return stage

    def training_epoch_end(self, outputs: List[Any]):
        if self.hparams.enable_log_prototypes and isinstance(self.logger, TensorBoardLogger):
            tb_logger = self.logger.experiment

            tb_logger.add_histogram(
                "train/prototypes", self.net.prototype_vectors, global_step=self.trainer.global_step,
            )

        return super().training_epoch_end(outputs)

    def validation_step(self, batch: BatchWithLabelType, batch_idx: int):
        logits, similarities, occurrences, nam_terms = self.forward(batch)

        targets = batch[-1]
        loss, preds = self._classification_loss(logits, targets)

        self._update_validation_metrics(loss, preds, targets)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        # compute balanced accuracy
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.val_cmat)

        cur_stage = self._current_stage
        # every 10th epoch is a last layer optim epoch
        if cur_stage == "nam_only":
            self.val_bacc_save.update(bacc)
            saver = bacc
        else:
            self.val_bacc_save.update(torch.tensor(0., dtype=torch.float32))
            saver = torch.tensor(-float('inf'), dtype=torch.float32, device=bacc.device)
        self.log("val/bacc_save_monitor", self.val_bacc_save.compute(), on_epoch=True)
        self.log("val/bacc_save", saver)

        self._log_validation_metrics()

    def test_step(self, batch: BatchWithLabelType, batch_idx: int):
        logits, similarities, occurrences, nam_terms = self.forward(batch)

        targets = batch[-1]
        loss, preds = self._classification_loss(logits, targets)

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

        prototypes = {
            'params': [self.net.prototype_vectors],
            'lr': self.hparams.lr,
            'weight_decay': 0.0,
        }
        nam = {name: p for name, p in self.net.nam.named_parameters() if p.requires_grad}
        embeddings = [nam.pop('tab_missing_embeddings')]
        embeddings = {
            'params': embeddings,
            'lr': self.hparams.lr,
            'weight_decay': 0.0,
        }
        nam = {
            'params': list(nam.values()),
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay_nam,
        }
        encoder = {
            'params': [p for p in self.net.features.parameters() if p.requires_grad],
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay,
        }
        occurrence_and_features = {
            'params': [p for p in self.net.add_on_layers.parameters() if p.requires_grad] +
            [p for p in self.net.occurrence_module.parameters() if p.requires_grad],
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay,
        }
        if len(encoder['params']) == 0:
            warnings.warn("Encoder seems to be frozen! No parameters require grad.")
        assert len(nam['params']) > 0
        assert len(prototypes['params']) > 0
        assert len(embeddings['params']) > 0
        assert len(occurrence_and_features['params']) > 0
        optim_all = torch.optim.AdamW([
            encoder, occurrence_and_features, prototypes, nam, embeddings
         ])
        optim_nam = torch.optim.AdamW([
            nam, embeddings
        ])
        optim_warmup = torch.optim.AdamW([
            occurrence_and_features, prototypes
        ])
        optim_warmup_protonet = torch.optim.AdamW([
            encoder, occurrence_and_features, prototypes
        ])

        training_iterations = len(self.trainer.datamodule.train_dataloader())
        LOG.info("Number of iterations for one epoch: %d", training_iterations)

        # stepping through warmup_protonet is sufficient, as all parameters groups are also in warmup
        scheduler_kwargs = {'max_lr': self.hparams.lr, 'cycle_momentum': False}
        scheduler_warmup = torch.optim.lr_scheduler.CyclicLR(
            optim_warmup_protonet,
            base_lr=self.hparams.lr / 20,
            max_lr=self.hparams.lr / 10,
            step_size_up=self.hparams.epochs_warmup * training_iterations,
            cycle_momentum=False
        )
        scheduler_all = torch.optim.lr_scheduler.CyclicLR(
            optim_all,
            base_lr=self.hparams.lr / 10,
            step_size_up=(self.hparams.epochs_all * training_iterations) / 2,
            step_size_down=(self.hparams.epochs_all * training_iterations) / 2,
            **scheduler_kwargs
        )
        scheduler_nam = torch.optim.lr_scheduler.CyclicLR(
            optim_nam,
            base_lr=self.hparams.lr / 10,
            step_size_up=(self.hparams.epochs_nam * training_iterations) / 2,
            step_size_down=(self.hparams.epochs_nam * training_iterations) / 2,
            **scheduler_kwargs
        )

        return ([optim_warmup, optim_warmup_protonet, optim_all, optim_nam],
                [scheduler_warmup, scheduler_all, scheduler_nam])

    def push_prototypes(self):
        LOG.info("Pushing protoypes. epoch=%d, step=%d", self.current_epoch, self.trainer.global_step)

        self.net.eval()

        prototype_shape = self.net.prototype_shape
        n_prototypes = self.net.num_prototypes

        global_max_proto_dist = np.full(n_prototypes, np.NINF)
        # global_max_proto_dist = np.ones(n_prototypes) * -1
        global_max_fmap = np.zeros(prototype_shape)
        global_img_indices = np.zeros(n_prototypes, dtype=np.int)
        global_img_classes = - np.ones(n_prototypes, dtype=np.int)

        if self.hparams.monitor_prototypes:
            proto_epoch_dir = Path(self.trainer.log_dir) / f"prototypes_epoch_{self.current_epoch}"
        else:
            proto_epoch_dir = Path(self.trainer.log_dir) / "prototypes_best"
        if self.hparams.enable_checkpointing:
            proto_epoch_dir.mkdir(exist_ok=True)

        push_dataloader = self.trainer.datamodule.push_dataloader()
        search_batch_size = push_dataloader.batch_size

        num_classes = self.net.num_classes

        save_embedding = self.hparams.enable_save_embeddings and isinstance(self.logger, TensorBoardLogger)

        # indicates which class a prototype belongs to
        proto_class = torch.argmax(self.net.prototype_class_identity, dim=1).detach().cpu().numpy()

        embedding_data = []
        embedding_labels = []
        for push_iter, (search_batch_input, _, _, search_y) in enumerate(push_dataloader):

            start_index_of_search_batch = push_iter * search_batch_size

            feature_vectors = self.update_prototypes_on_batch(
                search_batch_input,
                start_index_of_search_batch,
                global_max_proto_dist,
                global_max_fmap,
                global_img_indices,
                global_img_classes,
                num_classes,
                search_y,
                proto_epoch_dir,
            )

            if save_embedding:
                # for each batch, split into one feature vector for each prototype
                embedding_data.extend(feature_vectors[:, j] for j in range(n_prototypes))
                embedding_labels.append(np.repeat(proto_class, feature_vectors.shape[0]))

        prototype_update = np.reshape(global_max_fmap, tuple(prototype_shape))

        if self.hparams.enable_checkpointing:
            np.save(proto_epoch_dir / f"p_similarities_{self.current_epoch}.npy", global_max_proto_dist)
            np.save(proto_epoch_dir / f"p_feature_maps_{self.current_epoch}.npy", global_max_fmap)
            np.save(proto_epoch_dir / f"p_inp_indices_{self.current_epoch}.npy", global_img_indices)
            np.save(proto_epoch_dir / f"p_inp_img_labels_{self.current_epoch}.npy", global_img_classes)

        if save_embedding:
            tb_logger = self.logger.experiment

            embedding_data.append(self.net.prototype_vectors.detach().cpu().numpy())
            embedding_data.append(prototype_update)
            embedding_labels = np.concatenate(embedding_labels)
            metadata = [f"FV Class {i}" for i in embedding_labels]

            metadata.extend(f"Old PV Class {i}" for i in proto_class)
            metadata.extend(f"New PV Class {i}" for i in proto_class)

            tb_logger.add_embedding(
                mat=np.concatenate(embedding_data, axis=0),
                metadata=metadata,
                global_step=self.trainer.global_step,
                tag="push_prototypes",
            )

        self.net.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32, device=self.device))

        self.net.train()

    def update_prototypes_on_batch(
        self,
        search_batch,
        start_index_of_search_batch,
        global_max_proto_dist,
        global_max_fmap,
        global_img_indices,
        global_img_classes,
        num_classes,
        search_y,
        proto_epoch_dir,
    ):
        self.net.eval()

        feats, sims, occ = self.net.push_forward(
            search_batch[self.image_modality].to(self.device),
            search_batch[ModalityType.TABULAR].to(self.device))

        feature_vectors = np.copy(feats.detach().cpu().numpy())
        similarities = np.copy(sims.detach().cpu().numpy())
        occurrences = np.copy(occ.detach().cpu().numpy())

        del feats, sims, occ

        class_to_img_index = {key: [] for key in range(num_classes)}
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index[img_label].append(img_index)

        prototype_shape = self.net.prototype_shape
        n_prototypes = prototype_shape[0]

        for j in range(n_prototypes):
            target_class = torch.argmax(self.net.prototype_class_identity[j]).item()
            if len(class_to_img_index[target_class]) == 0:  # none of the images belongs to the class of this prototype
                continue
            proto_dist_j = similarities[class_to_img_index[target_class], j]
            # distnces of all latents to the j-th prototype of this class within the batch

            batch_max_proto_dist_j = np.amax(proto_dist_j)  # minimum distance of latents of this batch to prototype j

            if batch_max_proto_dist_j > global_max_proto_dist[j]:  # save if a new min distance is present in this batch

                img_index_in_class = np.argmax(proto_dist_j)
                img_index_in_batch = class_to_img_index[target_class][img_index_in_class]

                batch_max_fmap_j = feature_vectors[img_index_in_batch, j]

                # latent vector of minimum distance
                global_max_proto_dist[j] = batch_max_proto_dist_j
                global_max_fmap[j] = batch_max_fmap_j
                global_img_indices[j] = img_index_in_batch + start_index_of_search_batch
                global_img_classes[j] = search_y[img_index_in_batch].item()

                if self.hparams.enable_checkpointing:

                    # original image
                    original_img_j = search_batch[self.image_modality][img_index_in_batch].detach().cpu().numpy()

                    # find highly activated region of the original image
                    proto_occ_j = occurrences[img_index_in_batch, j]

                    np.save(proto_epoch_dir / f"original_{j}_epoch_{self.current_epoch}.npy", original_img_j)
                    np.save(proto_epoch_dir / f"occurrence_{j}_epoch_{self.current_epoch}.npy", proto_occ_j)

        return feature_vectors
