import torch
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet
from metrics import evaluate_model

def create_model():
    config = get_efficientdet_config("tf_efficientdet_d0")
    model = EfficientDet(config)
    model.class_net = HeadNet(config, num_outputs=7)
    return model

def train(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model().to(device)
    model.eval()

    metrics = evaluate_model(model, dataloader, device)

    print({
        "Model": "EfficientDet-D0",
        **metrics
    })
