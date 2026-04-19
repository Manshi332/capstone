# Real-Time ASL Recognition and Comparative Study of CNN and Transfer Learning Models

**A deep learning capstone project that recognises American Sign Language (ASL) hand gestures in real time using a MediaPipe skeleton pipeline, and systematically compares a custom-built CNN against three transfer learning backbones (ResNet50, VGG16, MobileNetV2) across 26 classes (A–Z).**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Why Skeleton Images?](#why-skeleton-images)
4. [Dataset](#dataset)
5. [Models](#models)
6. [Results](#results)
7. [Why These Results?](#why-these-results)
8. [Project Structure](#project-structure)
9. [Setup and Installation](#setup-and-installation)
10. [Usage](#usage)
11. [Training on Google Colab](#training-on-google-colab)
12. [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

This project has two distinct parts:

**Part 1 — Real-Time Webcam Recognition**
A live ASL recognition system that uses a webcam to capture hand signs and predicts the corresponding ASL letter in real time. The system uses MediaPipe to detect hand landmarks and renders a skeleton image, which is then fed to the trained model. This pipeline ensures the input at inference time matches the training domain exactly.

**Part 2 — Comparative Study of CNN Architectures**
A systematic comparison of four deep learning models on the same ASL skeleton dataset: a custom CNN built from scratch, and three transfer learning models (ResNet50, VGG16, MobileNetV2) with pretrained ImageNet weights. The study evaluates accuracy, precision, recall, F1-score, and real-world generalisation on unseen hand images.

---

## System Architecture

```
Raw webcam frame / photo
        │
        ▼
MediaPipe hand landmark detection (21 landmarks)
        │
        ▼
Skeleton renderer (colored joints on black background, 64×64)
        │
        ▼
Model inference (Custom CNN / ResNet50 / VGG16 / MobileNetV2)
        │
        ▼
Predicted ASL letter + confidence score
```

The key insight is the **skeleton rendering step**. Instead of feeding raw camera images directly to the model, MediaPipe detects 21 hand landmarks and renders them as a coloured skeleton on a black background. This has several advantages:

- **Background invariance**: The model never sees the room, lighting, or skin tone — only the hand geometry.
- **Domain consistency**: Training data and inference data are in exactly the same format.
- **Robustness**: Works across different skin tones, lighting conditions, and camera qualities.

---

## Why Skeleton Images?

The original approach of using raw cropped hand photos failed because models trained on one person's hand, one background, and one lighting setup would not generalise to different conditions. When tested with a stock photo of the Y sign, models trained on raw photos predicted R or W with low confidence.

After switching to skeleton images, the same Y sign photo was correctly predicted by models that had converged properly. The custom CNN achieved Y at 82.0% confidence, confirming that the skeleton domain abstraction is the right approach for a generalised ASL recognition system.


---

## Dataset

### Custom Webcam Dataset
- Collected using `src/collect_data.py` with MediaPipe landmark detection
- **7,500 images** — 300 images per letter for A–I, K–X, Y (24 letters); 150 images for J and Z
- J and Z are motion letters (J traces a J shape; Z traces a Z shape in the air) and are therefore under-represented in the custom set as only static frames can be captured
- Captured under controlled conditions: single person, single background

### Extra ASL Skeleton Dataset
- **59,476 images** — all 26 letters including J and Z
- Real hand photos converted to skeleton renders using `src/preprocess_to_skeleton.py`
- Provides diversity in hand shapes, proportions, and signing styles
- J and Z skeletons represent the mid-point of the motion gesture

### Why Two Datasets?
The custom dataset alone (7,500 images, one person) would cause the model to memorise a single person's hand geometry. The extra dataset introduces diversity in hand shape and size, improving generalisation. The two datasets are combined with a `WeightedRandomSampler` so that every class is sampled equally during training, preventing the larger extra dataset from dominating.

### Combined Training Configuration

| Split | Custom | Extra | Total |
|---|---|---|---|
| Train | 5,250 | 41,633 | 46,883 |
| Val | 1,125 | 8,921 | 10,046 |
| Test | 1,125 | 8,922 | 10,047 |

Both datasets are split independently using the same random seed (42) before combining, ensuring no data leakage between splits.

### Data Augmentation (Training Only)
Only geometric augmentations are applied to skeleton images. Color-based augmentations are deliberately excluded because the joint colors encode finger identity — destroying them would confuse the model.

| Augmentation | Reason |
|---|---|
| Random rotation ±25° | Hand tilt variation |
| Random affine (translate, scale, shear) | Position and size variation |
| Random perspective | Camera angle variation |
| Random erasing | Partial occlusion robustness |
| Random horizontal flip | Mirror-image hands |


---

## Models

### Custom CNN (Built from Scratch)

A 4-block convolutional network designed specifically for 64×64 skeleton images.

```
Input (3 × 64 × 64)
→ ConvBlock(3→32, dropout=0.10)     → 32×32
→ ConvBlock(32→64, dropout=0.15)    → 16×16
→ ConvBlock(64→128, dropout=0.20)   →  8×8
→ ConvBlock(128→256, dropout=0.20)  →  4×4
→ GlobalAveragePool → 256-d vector
→ FC(256→256) → BN → ReLU → Dropout(0.5)
→ FC(256→26)
```

Each ConvBlock contains Conv → BatchNorm → ReLU (×2 or ×3) → MaxPool → Dropout.

**Parameters**: ~2M

### Transfer Learning Models

All three pretrained on ImageNet, with frozen backbone initially, then fine-tuned from epoch 5.

| Model | Backbone params | Trainable after fine-tune | Head architecture |
|---|---|---|---|
| MobileNetV2 | 2.2M | 2.5M total | Dropout → FC(1280→256) → ReLU → Dropout → FC(256→26) |
| ResNet50 | 23.5M | 24.6M total | Dropout → FC(2048→512) → ReLU → BN → Dropout → FC(512→26) |
| VGG16 | 14.7M | 27.7M total | Dropout → FC(25088→512) → ReLU → BN → Dropout → FC(512→256) → ReLU → Dropout → FC(256→26) |

**Training strategy**: Backbone frozen for first 4 epochs (head-only training), then top layers unfrozen at epoch 5 with learning rate divided by 10.

---

## Results

### Test Set Performance (combined test split — 10,047 images: 1,125 custom + 8,922 extra)

| Model | Accuracy | Precision | Recall | F1 Weighted | F1 Macro |
|---|---|---|---|---|---|
| Custom CNN | **97.31%** | **97.33%** | **97.31%** | **97.30%** | **97.29%** |
| ResNet50 | **96.88%** | **96.94%** | **96.88%** | **96.88%** | **96.90%** |
| VGG16 | 93.53% | 93.62% | 93.53% | 93.52% | 93.47% |
| MobileNetV2 | 91.00% | 91.12% | 91.00% | 90.97% | 91.13% |


### Per-Class F1 Highlights



### Real-World Generalisation (unseen Y sign stock photo)

| Model | Predicted | Confidence |
|---|---|---|
| Custom CNN | ✓ Y | 82.05% |
| ResNet50 | ✓ Y | **90.94%** |
| VGG16 | ✓ Y | 19.93% |
| MobileNetV2 | ✓ Y | 26.06% |


---

## Why These Results?

### Why does the Custom CNN generalise better to the unseen photo despite lower test accuracy?


The test set is drawn from the same skeleton rendering pipeline and dataset split as the training data. Both models score ~99% because the test images come from the same distribution they were trained on. The real difference emerges on truly unseen images.

On the unseen Y photo, ResNet50 achieves higher confidence (90.94%) than the custom CNN (82.05%). This is the expected transfer learning advantage: ImageNet pretrained features encode rich spatial pattern detectors that generalise better across different hands, angles, and image sources.

### Why did VGG16 and MobileNetV2 perform worse on the real-world image despite reasonable test accuracy?

Both models had truncated training histories — MobileNetV2 was cut short mid-epoch and VGG16 converged to only 93.9% val accuracy compared to ResNet50's 97.2%. A partially trained transfer model has not fully adapted its backbone features to the skeleton domain, so it is less confident on unseen images. ResNet50's deeper residual connections and earlier fine-tuning (epoch 5 vs epoch 10) gave it more time to specialise.

### Why use a skeleton representation instead of raw photos?

Raw photo models fail in new environments because they co-learn the background, lighting, and skin tone alongside the hand shape. Skeleton images strip all non-geometric information, forcing the model to learn only from joint positions and connections. This is a form of structured feature extraction before model input, analogous to using keypoints for pose estimation rather than raw pixel data.

### Is the Custom CNN overfitting?


Not in the classical sense. There is no data leakage — both train and evaluate use the same seed-42 split, so the test set is genuinely held out. The 99.54% is a clean result. However, the CNN exhibits domain memorisation: it performs best on skeleton images rendered from the same dataset but shows lower confidence (82% vs ResNet50's 91%) on images rendered from a completely different source photo. This is the key distinction between memorisation and generalisation, and it is why ResNet50 is the recommended model for deployment.


---

## Project Structure

```
asl_recognition/
├── data/
│   ├── custom/          # Your webcam skeleton images (A-Z folders)
│   └── extra_skeleton/  # Extra ASL dataset converted to skeletons
├── models/              # Saved .pth checkpoints
├── results/             # Metrics, plots, confusion matrices
├── src/
│   ├── dataset.py           # Dataset loaders, augmentation, weighted sampler
│   ├── custom_cnn.py        # Custom CNN architecture
│   ├── transfer_model.py    # MobileNetV2, ResNet50, EfficientNet, VGG16
│   ├── train.py             # Training loop (Colab + local)
│   ├── evaluate.py          # Evaluation: accuracy, precision, recall, F1
│   ├── predict.py           # Single image prediction with skeleton rendering
│   ├── webcam.py            # Real-time webcam recognition
│   ├── collect_data.py      # Webcam data collection tool
│   ├── preprocess_to_skeleton.py  # Convert real photos to skeleton renders
│   └── compare.py           # Multi-model comparison charts
├── notebooks/
│   └── colab_training.ipynb # Google Colab training notebook
└── requirements.txt
```

---
## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/manshi332/asl-recognition.git
cd asl-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install mediapipe==0.10.14  
```

---

## Usage

### Collect Your Own Data
```bash
python src/collect_data.py --target_per_class 150 --output_dir data/custom
```
Press a letter key to start capturing. The script auto-saves every 0.15 seconds. Press ESC to quit.

### Convert Extra Dataset to Skeletons
```bash
python src/preprocess_to_skeleton.py \
    --input_dir data/extra_real \
    --output_dir data/extra_skeleton \
    --draw_style rich
```

### Train
```bash
# Custom CNN
python src/train.py --model cnn --source combined \
    --data_dir data/custom --extra_dir data/extra_skeleton \
    --epochs 20 --batch_size 32 --lr 0.001 \
    --save_path models/custom_cnn.pth

# ResNet50
python src/train.py --model transfer --backbone resnet50 \
    --source combined \
    --data_dir data/custom --extra_dir data/extra_skeleton \
    --epochs 20 --fine_tune_epoch 5 --batch_size 32 --lr 0.0003 \
    --save_path models/resnet50.pth
```

### Evaluate
```bash
python src/evaluate.py \
    --model_cnn models/custom_cnn.pth \
    --model_transfer models/resnet50.pth \
    --backbone resnet50 \
    --source combined \
    --data_dir data/custom \
    --extra_dir data/extra_skeleton

python src/compare.py --results_dir results --models_dir models
```

### Single Image Prediction
```bash
python src/predict.py \
    --image hand.jpg \
    --model models/resnet50.pth \
    --model_type transfer \
    --backbone resnet50 \
    --save_plot results/prediction.png
```

### Real-Time Webcam
```bash
python src/webcam.py \
    --model models/custom_cnn.pth \
    --model_type cnn
```

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
mediapipe==0.10.14
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=10.0.0
tqdm>=4.65.0
tensorboard>=2.13.0
```

---
