import torch
import torchvision
from metrics import evaluate_model

def get_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    return model

def train(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(7).to(device)
    model.eval()

    metrics = evaluate_model(model, dataloader, device)

    print({
        "Model": "MaskRCNN",
        **metrics
    })
