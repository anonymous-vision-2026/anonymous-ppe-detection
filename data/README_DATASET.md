# Dataset Description

This project uses a dataset for PPE detection in industrial environments.

The dataset annotations are available in two formats:

- YOLO format
- COCO format

The full dataset is not included due to size and privacy constraints.

---

## Classes

The dataset contains 7 classes:

- class_0
- class_1
- class_2
- class_3
- class_4
- class_5
- class_6

---

## Expected Structure (Full Dataset)

data/
 ├── train/images
 ├── train/labels
 ├── valid/images
 ├── valid/labels
 ├── test/images
 ├── test/labels

---

## YOLO Format

Each image has a corresponding `.txt` file.

Format:

class_id x_center y_center width height

Example:

0 0.52 0.48 0.30 0.40
2 0.25 0.60 0.10 0.15

---

## COCO Format

Annotations are stored in a single `.json` file following COCO standard:

- images
- annotations
- categories

Example structure:

{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}

---

## Sample Data

This repository includes:

- Sample images (`sample_images/`)
- Sample YOLO labels (`sample_labels_yolo/`)
- Sample COCO annotations (`sample_annotations_coco/`)

These are provided only for illustration purposes.
