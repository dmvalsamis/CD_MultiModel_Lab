from typing import Any, Callable, Optional, Union, Sequence

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.base.model import SegmentationModel
from torch import Tensor

class FCSiamDiff(SegmentationModel):
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
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
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
        # Add a dummy classification head to satisfy the base class initialization
        self.classification_head = None
        self.name = f"u-{encoder_name}"
        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        features1, features2 = self.encoder(x1), self.encoder(x2)
        features = [features2[i] - features1[i] for i in range(1, len(features1))]
        features.insert(0, features2[0])
        decoder_output = self.decoder(*features)
        masks: Tensor = self.segmentation_head(decoder_output)
        return masks


# Model Evaluation

device = torch.device("cpu")
model = FCSiamDiff().to(device)

# Load the trained model
model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/FCSiamDiff_CBMI_6acc.h5'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Example input
dummy_input1 = torch.randn(1, 3, 96, 96).to(device)
dummy_input2 = torch.randn(1, 3, 96, 96).to(device)
input_tensor = torch.stack((dummy_input1, dummy_input2), dim=1)

# Forward pass
with torch.no_grad():
    output = model(input_tensor)

print(output.shape)
