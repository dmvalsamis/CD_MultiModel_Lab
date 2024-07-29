from typing import Any, Callable, Optional, Union, Sequence
import segmentation_models_pytorch as smp
import torch
from torch import Tensor
from segmentation_models_pytorch.base.model import SegmentationModel

class FCSiamConc(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
    ):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        encoder_out_channels = [c * 2 for c in self.encoder.out_channels[1:]]
        encoder_out_channels.insert(0, self.encoder.out_channels[0])
        try:
            UnetDecoder = smp.decoders.unet.decoder.UnetDecoder
        except AttributeError:
            UnetDecoder = smp.unet.decoder.UnetDecoder
        self.decoder = UnetDecoder(
            encoder_channels=encoder_out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        self.classification_head = None
        self.name = f"u-{encoder_name}"
        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        print("X1: ", x1.shape)
        print("X2: ", x2.shape)
        features1, features2 = self.encoder(x1), self.encoder(x2)
        
        for i in range(len(features1)):
            print(f"Features1[{i}]: {features1[i].shape}")
            print(f"Features2[{i}]: {features2[i].shape}")
            print(f"Features1[{i}] min: {features1[i].min()}, max: {features1[i].max()}")
            print(f"Features2[{i}] min: {features2[i].min()}, max: {features2[i].max()}")

        features = [features2[0]]  
        for i in range(1, len(features1)):
            combined = torch.cat([features2[i], features1[i]], dim=1)
            print(f"Combined feature at level {i}: {combined.shape}")
            features.append(combined)

        try:
            decoder_output = self.decoder(*features)
            print("Decoder output shape: ", decoder_output.shape)
        except RuntimeError as e:
            print("RuntimeError in decoder: ", str(e))
            for i, f in enumerate(features):
                print(f"Feature {i} shape at error: {f.shape}")
            raise e  

        masks: Tensor = self.segmentation_head(decoder_output)
        return masks
