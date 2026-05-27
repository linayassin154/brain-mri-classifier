# Brain Tumor Classification

Deep learning project for classifying brain MRI scans as tumor or no tumor using transfer learning and image preprocessing techniques.

**Dataset:** [Br35H Brain Tumor Detection](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

---

## What This Project Includes

### Image Preprocessing
- Non-local means denoising
- CLAHE contrast enhancement in LAB color space
- Saved before/after preprocessing visualizations

### Data Pipeline
- TensorFlow `tf.data` pipeline
- Shuffle, batching, and prefetching
- Data augmentation:
  - Horizontal flip
  - Rotation
  - Zoom
  - Brightness & contrast adjustments
- Class weight balancing for imbalanced classes

---

## Models

Transfer learning with:
- ResNet50V2
- EfficientNetB0

Custom classification head:
- GlobalAveragePooling
- Dense layer
- Batch Normalization
- ReLU activation
- Dropout
- Sigmoid output

---

## Training
- Learning rate and batch size experiments
- Two-stage training:
  1. Frozen backbone training
  2. Fine-tuning the final layers
- Callbacks used:
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint

---

## Evaluation
- Accuracy, Precision, Recall, F1-score, and AUC-ROC
- Confusion matrix and ROC curve visualization
- Correct vs incorrect prediction samples
- Metrics and training history exported to JSON

---

## Deployment
Includes a Gradio web app where users can upload an MRI scan and receive a prediction with a confidence score.

---

## Tech Stack
Python, TensorFlow/Keras, OpenCV, scikit-learn, NumPy, Matplotlib, Gradio
