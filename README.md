# Real-Time ASL Recognition and Comparative Study of CNN and Transfer Learning Models

**A deep learning capstone project that recognises American Sign Language (ASL) hand gestures in real time using a MediaPipe skeleton pipeline, and systematically compares a custom-built CNN against three transfer learning backbones (ResNet50, VGG16, MobileNetV2) across 26 classes (A–Z).**


## Project Overview

This project has two distinct parts:

**Part 1 — Real-Time Webcam Recognition**
A live ASL recognition system that uses a webcam to capture hand signs and predicts the corresponding ASL letter in real time. The system uses MediaPipe to detect hand landmarks and renders a skeleton image, which is then fed to the trained model. This pipeline ensures the input at inference time matches the training domain exactly.

**Part 2 — Comparative Study of CNN Architectures**
A systematic comparison of four deep learning models on the same ASL skeleton dataset: a custom CNN built from scratch, and three transfer learning models (ResNet50, VGG16, MobileNetV2) with pretrained ImageNet weights. The study evaluates accuracy, precision, recall, F1-score, and real-world generalisation on unseen hand images.

---

## Dataset

### Custom Webcam Dataset
- Collected using `src/collect_data.py` with MediaPipe landmark detection
- **7,500 images** — 300 images per letter for A-Z


### Extra ASL Skeleton Dataset
- **59,476 images** — all 26 letters 
- Real hand photos converted to skeleton images using `src/preprocessingn.py`


### Why Two Datasets?
The custom dataset alone (7,500 images, one person) would cause the model to memorise a single person's hand geometry. The extra dataset introduces diversity in hand shape and size, improving generalisation. The two datasets are combined with a `WeightedRandomSampler` so that every class is sampled equally during training, preventing the larger extra dataset from dominating.

### Why Skeleton Images?

The original approach of using raw cropped hand photos failed because models trained on one person's hand, one background, and one lighting setup would not generalise to different conditions. When tested with a stock photo of the Y sign, models trained on raw photos predicted R or W with low confidence.

After switching to skeleton images, the same Y sign photo was correctly predicted:
- Custom CNN: **Y at 82.05%**
- ResNet50: **Y at 90.94%**

This confirms that the skeleton domain abstraction is the right approach for a generalised ASL recognition system

### Combined Training Configuration

| Split | Custom | Extra | Total |
|---|---|---|---|
| Train | 5,250 | 41,633 | 46,883 |
| Val | 1,125 | 8,921 | 10,046 |
| Test | 1,125 | 8,922 | 10,047 |

Both datasets are split independently using the same random seed (42) before combining, ensuring no data leakage between splits.

## Results

### Test Set Performance (combined test split — 10,047 images: 1,125 custom + 8,922 extra)

| Model | Accuracy | Precision | Recall | F1 Weighted | F1 Macro |
|---|---|---|---|---|---|
| Custom CNN | **97.31%** | **97.33%** | **97.31%** | **97.30%** | **97.29%** |
| ResNet50 | **96.88%** | **96.94%** | **96.88%** | **96.88%** | **96.90%** |
| VGG16 | 93.53% | 93.62% | 93.53% | 93.52% | 93.47% |
| MobileNetV2 | 91.00% | 91.12% | 91.00% | 90.97% | 91.13% |

### Real-World Generalisation (unseen Y sign stock photo)

| Model | Predicted | Confidence |
|---|---|---|
| Custom CNN |  Y | 82.05% |
| ResNet50 |  Y | **90.94%** |
| VGG16 |  Y | 19.93% |
| MobileNetV2 |  Y | 26.06% |


<img width="2560" height="581" alt="image" src="https://github.com/user-attachments/assets/433757e1-5990-4726-875e-35cba4140bed" />
<img width="2559" height="581" alt="image" src="https://github.com/user-attachments/assets/6539e07a-ff1e-4847-8756-2457f89ea2c6" />
<img width="2561" height="581" alt="image" src="https://github.com/user-attachments/assets/763ac5d4-d95d-411f-a44a-997c2188a299" />
<img width="2561" height="581" alt="image" src="https://github.com/user-attachments/assets/6a74caff-22af-49c4-ad7d-344eceb90ba3" />




