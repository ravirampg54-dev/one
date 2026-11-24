# Goat Eye Anemia (FAMACHA) Screening — ML Pipeline

An industrial-style, lightweight (no-GPU-required) pipeline that:
1. Dynamically locates the goat's **eye** in any photo (no manual cropping needed).
2. Dynamically segments **only the FAMACHA conjunctiva strip** (lower inner eyelid) — the exact region vets use for anemia scoring.
3. Extracts color/texture features that quantify pallor.
4. Trains and compares **SVM, KNN, and Random Forest**, picks the best via cross-validation.
5. Serves a Streamlit web app matching the "Goat Eye Health Screening" UI, for live upload → analyze → result.

## Why this architecture (and not deep learning)

For a farm-deployment "industrial project," classical ML on hand-crafted color features is the right call:
- Works on **CPU-only devices** (no GPU needed in the field).
- Needs far **less training data** (tens–hundreds of images vs. thousands for CNNs).
- Fully **explainable** — you can literally show a vet the exact color/texture numbers that drove the decision.
- Trains in **seconds**, so you can retrain per goat breed/region easily.

If you later collect 1,000+ labeled images, this same segmentation step feeds cleanly into a CNN (e.g., MobileNetV3) for a further accuracy boost — the segmentation module doesn't change.

## Project structure

```
goat_anemia_project/
├── data/
│   └── raw/
│       ├── healthy/     <- put normal-conjunctiva images here
│       └── anemic/      <- put pale-conjunctiva images here
├── src/
│   ├── conjunctiva_segmentation.py   # dynamic eye + FAMACHA-region detection
│   ├── feature_extraction.py         # 34-dim color/texture feature vector
│   ├── train_model.py                # trains + compares SVM/KNN/RandomForest
│   └── predict.py                    # inference on a new image
├── app/
│   └── app.py                        # Streamlit demo UI
├── models/                           # saved model + scaler + metrics (after training)
├── outputs/                          # debug overlays
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Add your dataset

Drop your labeled images into:
```
data/raw/healthy/*.jpg
data/raw/anemic/*.jpg
```
No manual bounding boxes needed — the segmentation module finds the conjunctiva automatically. (If your existing dataset already has bounding-box annotations, `extract_conjunctiva()` accepts `manual_bbox=(x,y,w,h)` for higher precision — wire that in inside `train_model.py`'s `build_dataset()` if you have that metadata.)

## 2. Train

```bash
python src/train_model.py
```

This prints **honest**, cross-validated accuracy/precision/recall/F1/confusion-matrix for each of SVM, KNN, and Random Forest, and saves the best one to `models/`.

> **On "100% accuracy":** a real biological classifier reporting 100% almost always means it memorized a small/skewed training set and will fail on new goats. This script reports 5-fold CV + held-out test accuracy specifically so the number you get is trustworthy. Push accuracy up by adding more images across different goat breeds, ages, lighting conditions, and camera angles — not by relaxing the evaluation.

## 3. Predict on a new image

```bash
python src/predict.py path/to/new_goat_eye.jpg
```

## 4. Run the web app

```bash
streamlit run app/app.py
```

## Tuning for higher accuracy

- **More data, more diversity** is the #1 lever — aim for 150+ images per class minimum, from multiple goats/farms/lighting.
- Try `class_weight="balanced"` in SVM/RandomForest if your dataset is imbalanced.
- Add more FAMACHA classes (1–5 scale) instead of binary healthy/anemic once you have enough data per class, for finer-grained severity scoring.
- If accuracy plateaus below what's needed for production, the next step up is a lightweight CNN (MobileNetV3/EfficientNet-Lite) fine-tuned on the segmented conjunctiva crops — same segmentation module feeds it directly.
