# Create models script

import torch
import segmentation_models_pytorch as smp



def deeplabv3_plus(args) -> torch.nn.Module:
    
    model = smp.DeepLabV3Plus(
        encoder_name=args.encoder,
        encoder_depth=args.encoder_depth,
        encoder_weights=args.encoder_weights,
        in_channels=args.input_channels, 
        classes=args.num_classes
    )

    return model

def fpn(args) -> torch.nn.Module:

    model = smp.FPN(
        encoder_name=args.encoder,
        encoder_depth=args.encoder_depth,
        encoder_weights=args.encoder_weights,
        in_channels=args.input_channels,
        classes=args.num_classes
    )


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl_model = deeplabv3_plus().to(device)
    print(dl_model)


    img = torch.rand((10, 4, 512, 512), device=device)
    out = dl_model(img)

    print(out)
    print(out.shape)