import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from metrics import evaluate_model, measure_inference_time

def train_ssd(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ssdlite320_mobilenet_v3_large(pretrained=True)
    model.to(device)

    model.eval()

    metrics = evaluate_model(model, dataloader, device)
    inference = measure_inference_time(model, dataloader, device)

    print({
        "Model": "SSD-MobileNet",
        **metrics,
        "Inference_ms": inference
    })
