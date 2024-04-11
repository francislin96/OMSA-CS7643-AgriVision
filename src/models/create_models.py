# Create models script

import torch
import segmentation_models_pytorch as smp



def deeplabv3_plus(num_classes, **args) -> torch.nn.Module:
    
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_depth=5,
        encoder_weights=None,
        in_channels=4, 
        classes=num_classes
    )

    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl_model = deeplabv3_plus().to(device)
    print(dl_model)


    img = torch.rand((10, 4, 512, 512), device=device)
    out = dl_model(img)

    print(out)
    print(out.shape)