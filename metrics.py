import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time

def evaluate_model(model, dataloader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            preds = []
            gts = []

            for i in range(len(outputs)):
                preds.append({
                    "boxes": outputs[i]["boxes"].cpu(),
                    "scores": outputs[i]["scores"].cpu(),
                    "labels": outputs[i]["labels"].cpu()
                })

                gts.append({
                    "boxes": targets[i]["boxes"].cpu(),
                    "labels": targets[i]["labels"].cpu()
                })

            metric.update(preds, gts)

    results = metric.compute()

    return {
        "mAP@0.5": float(results["map_50"]),
        "mAP@0.5:0.95": float(results["map"]),
        "Recall": float(results["mar_100"])
    }


def measure_inference_time(model, dataloader, device):
    model.eval()
    total_time = 0
    n = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = [img.to(device) for img in images]

            start = time.time()
            model(images)
            end = time.time()

            total_time += (end - start)
            n += len(images)

    return (total_time / n) * 1000  # ms
