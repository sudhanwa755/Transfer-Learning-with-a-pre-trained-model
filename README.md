<div align="center">

# 🐱 vs 🐶 &nbsp; Cats vs Dogs Classifier

### Binary Image Classification with Transfer Learning & MobileNetV2

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-34A853?style=for-the-badge&logo=google&logoColor=white)](https://keras.io/api/applications/mobilenet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Train a high-accuracy image classifier in under 5 minutes** — using the power of pretrained ImageNet knowledge, applied to one of the most iconic machine learning challenges.

</div>

---

## 📌 Table of Contents

- [🌟 Project Overview](#-project-overview)
- [🧠 Core Concepts Explained](#-core-concepts-explained)
  - [Why Cats vs Dogs?](#why-cats-vs-dogs)
  - [Transfer Learning — Standing on the Shoulders of Giants](#transfer-learning--standing-on-the-shoulders-of-giants)
  - [MobileNetV2 — Small Footprint, Big Brains](#mobilenetv2--small-footprint-big-brains)
  - [The Data Pipeline — Map, Shuffle, Batch, Prefetch](#the-data-pipeline--map-shuffle-batch-prefetch)
  - [Frozen Weights — Preserving Pretrained Knowledge](#frozen-weights--preserving-pretrained-knowledge)
- [🏗️ Architecture Diagram](#-architecture-diagram)
- [📂 Project Structure](#-project-structure)
- [⚙️ Setup & Installation](#-setup--installation)
- [🚀 How to Run](#-how-to-run)
- [📊 Results](#-results)
- [🔬 Code Walkthrough](#-code-walkthrough)
- [🛠️ Configuration & Hyperparameters](#-configuration--hyperparameters)
- [🔭 Future Improvements](#-future-improvements)

---

## 🌟 Project Overview

This project demonstrates how to build a **production-grade binary image classifier** that distinguishes between cats and dogs — without training from scratch. Instead, we leverage **Transfer Learning** with **MobileNetV2** (pretrained on ImageNet), dramatically reducing training time and data requirements while maintaining high accuracy.

| Property | Value |
|---|---|
| **Task** | Binary Image Classification (Cat / Dog) |
| **Dataset** | `cats_vs_dogs` via TensorFlow Datasets |
| **Base Model** | MobileNetV2 (ImageNet weights) |
| **Training Strategy** | Transfer Learning (frozen base + custom head) |
| **Input Size** | 150 × 150 × 3 |
| **Epochs** | 5 |
| **Batch Size** | 64 |

---

## 🧠 Core Concepts Explained

### Why Cats vs Dogs?

At first glance, this seems simple — but it's deceptively challenging for a machine. Cats and dogs share similar visual attributes: fur texture, four limbs, similar proportions, variable backgrounds, and enormous diversity within each class (a chihuahua looks nothing like a husky). This makes the problem an ideal benchmark for evaluating a model's ability to learn **abstract, discriminative visual features** rather than memorizing simple patterns.

It also serves as a **gateway problem** — mastering this teaches you the same skills needed for medical imaging, satellite analysis, or any real-world classification task.

---

### Transfer Learning — Standing on the Shoulders of Giants

> *"Why build a library from scratch when the world's greatest library already exists?"*

Training a deep neural network from scratch requires:
- **Millions** of labeled images
- **Days or weeks** of GPU compute
- Extensive expertise in initialization, regularization, and architecture design

**Transfer Learning** bypasses all of this. The core idea:

```
Pre-trained Model (trained on 1.2M ImageNet images)
        │
        │  Already knows about:
        │   ✅ Edges & gradients
        │   ✅ Textures & patterns
        │   ✅ Shapes & object parts
        │   ✅ High-level semantic features
        ▼
Freeze the base layers  →  Add a tiny custom "head"  →  Train only the head
        │
        ▼
  New Classifier (Cats vs Dogs)
  Trained in minutes, not days!
```

The model doesn't learn "what a cat is" from zero — it already understands visual structure from ImageNet. We only teach it to **redirect that knowledge** toward our specific two-class problem.

This approach is not just faster — it often achieves **better accuracy** than training from scratch on small datasets, because the pretrained representations generalize extremely well.

---

### MobileNetV2 — Small Footprint, Big Brains

**MobileNetV2** is a convolutional neural network architecture designed by Google specifically for **speed and memory efficiency**, without sacrificing accuracy.

Key innovations that make it special:

| Feature | What It Does |
|---|---|
| **Depthwise Separable Convolutions** | Splits standard convolutions into two lighter operations, reducing computation by ~8–9× |
| **Inverted Residuals** | Expands feature channels in the middle of a block (rather than compressing), enabling richer representations at low cost |
| **Linear Bottlenecks** | Prevents information loss by using linear activations at bottleneck layers |
| **~3.4M Parameters** | Tiny compared to VGG16 (~138M) or ResNet50 (~25M) |

In our project, we use MobileNetV2 with `include_top=False`, which strips off the original ImageNet classification layers and gives us the raw feature extractor — a 4D tensor of learned visual features — ready to plug into our own classifier.

---

### The Data Pipeline — Map, Shuffle, Batch, Prefetch

Efficient data loading is just as important as model design. A slow pipeline starves your GPU. We build ours using TensorFlow's `tf.data` API with four key steps:

```
Raw Dataset
    │
    ▼
  MAP         →  Resize images to 150×150, normalize pixels to [0.0, 1.0]
    │              (Why normalize? Neural nets converge faster & more stably
    │               with inputs in a small, consistent range)
    ▼
 SHUFFLE      →  Randomize order within a buffer of 1000 samples
    │              (Prevents the model from learning order-based patterns
    │               and ensures varied batches for better generalization)
    ▼
  BATCH       →  Group into batches of 64
    │              (Single-sample updates are noisy; batching balances
    │               gradient stability and memory efficiency)
    ▼
 PREFETCH     →  CPU prepares batch N+1 while GPU trains on batch N
                   (Eliminates I/O bottleneck — the GPU is never idle waiting
                    for data. AUTOTUNE dynamically picks the buffer size.)
```

**Why `AUTOTUNE`?** Rather than hand-tuning prefetch buffer size, `tf.data.AUTOTUNE` profiles your system at runtime and automatically selects the optimal value. This is a best practice for production pipelines.

---

### Frozen Weights — Preserving Pretrained Knowledge

```python
base_model.trainable = False
```

This single line is one of the most important in the entire project. Here's why:

When we set `trainable = False`, the weights of MobileNetV2's convolutional layers are **locked** during our training run. Gradients still flow through them during backpropagation for calculation purposes, but the optimizer **does not update** those weights.

Without freezing:
- ❌ The pretrained knowledge gets overwritten ("catastrophic forgetting")
- ❌ Much more compute required
- ❌ Risk of overfitting on our small dataset subset

With freezing:
- ✅ We inherit 1.2M images worth of visual knowledge instantly
- ✅ Only the lightweight classification head (GlobalAveragePooling2D + Dense) is trained
- ✅ Dramatically fewer parameters to optimize → less data needed

---

## 🏗️ Architecture Diagram

```
Input Image (150 × 150 × 3)
         │
         ▼
┌─────────────────────────────────┐
│         MobileNetV2             │
│   (Pretrained on ImageNet)      │  ← FROZEN — weights not updated
│                                 │
│  Conv2D → BN → ReLU6            │
│  Depthwise Separable Blocks     │
│  Inverted Residual Blocks       │
│  ...                            │
│  Output: (5, 5, 1280) tensor    │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│    GlobalAveragePooling2D       │  ← Collapses spatial dims → (1280,)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│    Dense(1, activation='sigmoid')│  ← Output: probability [0.0, 1.0]
└─────────────────────────────────┘
         │
         ▼
    0.0 = Cat  /  1.0 = Dog
```

---

## 📂 Project Structure

```
cats-vs-dogs-classifier/
│
├── Transfer_Learning_with_a_pre_trained_model.ipynb   # Main notebook (model, training, visualization)
└── README.md                                          # You are here
```

---

## ⚙️ Setup & Installation

**Prerequisites:** Python 3.8+, pip, Jupyter Notebook or JupyterLab

```bash
# 1. Clone the repository
git clone https://github.com/your-username/cats-vs-dogs-classifier.git
cd cats-vs-dogs-classifier

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow tensorflow-datasets matplotlib numpy jupyter
```

---

## 🚀 How to Run

```bash
# Launch Jupyter Notebook
jupyter notebook Transfer_Learning_with_a_pre_trained_model.ipynb
```

Then run all cells in order via **Kernel → Restart & Run All**.

On first run, TensorFlow Datasets will automatically download the `cats_vs_dogs` dataset (~800 MB). Subsequent runs use the cached version.

**Expected output:**
```
Downloading and preparing dataset...
Starting training...
Epoch 1/5 - loss: 0.3421 - accuracy: 0.8512 - val_accuracy: 0.9104
Epoch 2/5 - loss: 0.1893 - accuracy: 0.9301 - val_accuracy: 0.9388
...
```

---

## 📊 Results

After just **5 epochs** on a 10% data subset:

| Metric | Training | Validation |
|---|---|---|
| **Accuracy** | ~93% | ~95% |
| **Loss** | ~0.17 | ~0.13 |

> 💡 These results demonstrate the extraordinary power of transfer learning — achieving **>90% accuracy with minimal data and compute**, which would be impossible training from scratch under the same conditions.

The script automatically generates training/validation accuracy and loss curves for visual analysis of model convergence.

---

## 🔬 Code Walkthrough

```python
# ── 1. CONFIGURATION ──────────────────────────────────────────────────
IMG_SIZE = 150      # Resize all images to 150×150 pixels
BATCH_SIZE = 64     # Process 64 images per gradient update
EPOCHS = 5          # Full passes through the training data

# ── 2. DATA LOADING ───────────────────────────────────────────────────
# Load 10% for training, last 5% for validation (fast demo subset)
(train_ds, test_ds), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:10%]', 'train[95%:]'],
    as_supervised=True,   # Returns (image, label) tuples
    with_info=True
)

# ── 3. PREPROCESSING ──────────────────────────────────────────────────
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))  # Standardize size
    return tf.cast(image, tf.float32) / 255.0, label      # Normalize to [0,1]

# ── 4. PIPELINE ───────────────────────────────────────────────────────
train_batches = (train_ds
    .map(preprocess)           # Apply preprocessing to each sample
    .shuffle(1000)             # Randomize order
    .batch(BATCH_SIZE)         # Group into batches
    .prefetch(tf.data.AUTOTUNE)) # Async prefetch for GPU efficiency

# ── 5. MODEL ──────────────────────────────────────────────────────────
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,         # Remove ImageNet classification head
    weights='imagenet'         # Load pretrained weights
)
base_model.trainable = False   # 🔒 Freeze — preserve learned features

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),   # Flatten feature maps
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary output
])

# ── 6. COMPILE & TRAIN ────────────────────────────────────────────────
model.compile(
    optimizer='adam',               # Adaptive learning rate optimizer
    loss='binary_crossentropy',     # Standard loss for binary classification
    metrics=['accuracy']
)

history = model.fit(train_batches, epochs=EPOCHS, validation_data=test_batches)
```

---

## 🛠️ Configuration & Hyperparameters

| Parameter | Default | Effect of Increasing |
|---|---|---|
| `IMG_SIZE` | `150` | Higher detail, more memory & compute |
| `BATCH_SIZE` | `64` | Smoother gradients, more memory required |
| `EPOCHS` | `5` | Better convergence (watch for overfitting) |
| Training split | `10%` | More data → better generalization |

To run on the **full dataset** for maximum accuracy, change the split:
```python
split=['train[:80%]', 'train[80%:]']
```

---

## 🔭 Future Improvements

- [ ] **Fine-tuning** — Unfreeze the top layers of MobileNetV2 after initial training for a second-stage boost
- [ ] **Data Augmentation** — Add random flips, rotations, zoom to improve robustness
- [ ] **Grad-CAM Visualization** — Highlight which image regions the model focuses on
- [ ] **Model Export** — Save as TensorFlow SavedModel or TFLite for mobile deployment
- [ ] **Web Demo** — Deploy with Gradio or Streamlit for an interactive UI

---

<div align="center">

Made with ❤️ and TensorFlow &nbsp;|&nbsp; Powered by Transfer Learning

*If this project helped you, consider giving it a ⭐ on GitHub!*

</div>
