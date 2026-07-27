# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ROTATED_DETECTORS
from .oriented_rcnn import OrientedRCNN


@ROTATED_DETECTORS.register_module()
class AFE(OrientedRCNN):
    """OrientedRCNN with auxiliary losses from AFE neck.

    Extends OrientedRCNN to automatically collect auxiliary losses
    (balance loss + diversity loss) from the neck during training.
    """

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        losses = super().forward_train(
            img, img_metas, gt_bboxes, gt_labels,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_masks=gt_masks, proposals=proposals, **kwargs)

        # Collect auxiliary losses from neck (balance loss + diversity loss)
        if self.with_neck and hasattr(self.neck, 'get_balance_loss'):
            aux = self.neck.get_balance_loss()
            if isinstance(aux, torch.Tensor) and aux > 0:
                losses['aux_balance_loss'] = aux

        return losses
