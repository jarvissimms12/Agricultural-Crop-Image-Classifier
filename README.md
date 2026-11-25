# 🌾 Agricultural Crop Image Classifier

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?logo=tensorflow)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Summary

| | |
|---|---|
| **Question** | Can transfer learning identify 30 crop types from ordinary photos with a small dataset? |
| **Data** | [Kaggle: Agricultural Crops Image Classification](https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification), 30 classes (cherry, coffee plant, cucumber, lemon, olive tree, rice, soybean, sugarcane, wheat…), split 80/10/10 with a fixed seed |
| **Method** | ImageNet ResNet50 → global average pooling → Dense(128) → Dropout(0.5) → 30-way softmax. Phase 1 trains the head on a frozen base; phase 2 fine-tunes the last ResNet stage (`conv5`) at a 1e-5 learning rate. Real-time augmentation, early stopping on validation accuracy. |
| **Result** | An earlier Kaggle run logged **81.9% test accuracy** (training ~96%, validation ~88%; see `Analysis.txt`). With 30 classes, chance is 3.3%. |
| **Takeaway** | A frozen pretrained backbone gets most of the way on a small dataset. The train/validation gap (~8 points) says more data or stronger augmentation would help more than a bigger model. |

### What changed in this version

The previously committed `main.py` couldn't reproduce the reported accuracy. It had a
syntax error (an unclosed parenthesis), never built a test set, never evaluated, trained
without validation data, and used a different head than the README described. The 81.9%
figure comes from the earlier Kaggle notebook run and is labeled that way above.

`main.py` is now a complete, runnable pipeline:

- train / validation / **held-out test** generators (test unshuffled, so predictions align)
- two-phase training: frozen head, then fine-tuning of `conv5`
- early stopping that restores the best validation weights
- **one** evaluation on the test split, written to `results/metrics.json` with top-1 and top-3 accuracy, plus a per-class precision/recall report and training curves

The pipeline was verified end to end on a small synthetic image set. Rerunning it on the
Kaggle data (GPU recommended) will regenerate `results/` and replace the earlier figure
with reproducible numbers.

## How to run

```bash
pip install -r requirements.txt
kaggle datasets download -d mdwaquarazam/agricultural-crops-image-classification
unzip agricultural-crops-image-classification.zip -d Agricultural-crops
python main.py --epochs-head 15 --epochs-finetune 10
```

Outputs: `results/metrics.json`, `results/per_class_report.txt`,
`results/training_curves.png`, `results/CropModel.keras`.

## Project structure

```
Agricultural-Crop-Image-Classifier/
├── main.py           # split → train head → fine-tune → evaluate on test → save
├── requirements.txt
├── Analysis.txt      # notes and figures from the earlier Kaggle run
└── README.md
```

## Skills demonstrated

Transfer learning and fine-tuning (ResNet50) · data augmentation · train/validation/test
discipline · per-class evaluation · reproducible CLI pipelines.

## Related projects

- [state-of-nyc](https://github.com/jarvissimms12/state-of-nyc): housing, schools and city jobs across NYC's 59 community districts
- [Credit-Card-Fraud](https://github.com/jarvissimms12/Credit-Card-Fraud): imbalanced classification evaluated on realistic data

---

**Jarvis Simms** | MS Data Science, NYIT 2025 | Brooklyn, NY ·
[GitHub](https://github.com/jarvissimms12) · [LinkedIn](https://www.linkedin.com/in/jarvis-simms-846062136/)
