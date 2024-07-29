from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.base.model import SegmentationModel
from torch import Tensor




class FCSiamDiff(smp.Unet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["aux_params"] = None
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        features1, features2 = self.encoder(x1), self.encoder(x2)
        features = [features2[i] - features1[i] for i in range(1, len(features1))]
        features.insert(0, features2[0])
        decoder_output = self.decoder(*features)
        masks: Tensor = self.segmentation_head(decoder_output)
        return masks