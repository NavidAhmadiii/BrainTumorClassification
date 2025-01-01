### **Brain Tumor Classification using Deep Learning** 🧠📊

This project aims to classify brain tumor images into different categories using a deep learning model. The model is trained on a dataset of brain tumor images and can classify them into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

---

## **Table of Contents** 📑
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)

---

## **Project Overview** 🌟
This project uses a **Convolutional Neural Network (CNN)** to classify brain tumor images. The model is trained on a dataset of brain MRI scans and can classify them into four categories. The goal is to assist medical professionals in diagnosing brain tumors more efficiently.

---

## **Dataset** 📂
The dataset contains brain MRI images divided into four categories:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

The dataset is split into three folders:
- **Training**: Images for training the model.
- **Validation**: Images for validating the model during training.
- **Test**: Images for testing the model after training.

---

## **Installation** ⚙️
To run this project, you need to have Python installed along with the following libraries:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain_tumor_classification.git
   cd brain_tumor_classification
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage** 🚀
1. **Prepare the Dataset:**
   - Place your dataset in the `data` folder with the following structure:
     ```
     data/
     ├── Training/
     │   ├── Glioma/
     │   ├── Meningioma/
     │   ├── Pituitary/
     │   └── No Tumor/
     ├── Validation/
     │   ├── Glioma/
     │   ├── Meningioma/
     │   ├── Pituitary/
     │   └── No Tumor/
     └── Test/
         ├── Glioma/
         ├── Meningioma/
         ├── Pituitary/
         └── No Tumor/
     ```

2. **Train the Model:**
   - Run the following command to train the model:
     ```bash
     python main.py
     ```

3. **Evaluate the Model:**
   - After training, the model will be evaluated on the test dataset, and the results will be displayed.

4. **Make Predictions:**
   - You can use the trained model to make predictions on new images by modifying the `predict.py` script.

---

## **Model Architecture** 🧠
The model architecture is a **Convolutional Neural Network (CNN)** with the following layers:
1. **Input Layer**: Accepts images of size 150x150x3.
2. **Convolutional Layers**: Three convolutional layers with ReLU activation.
3. **Max Pooling Layers**: Three max pooling layers to reduce dimensionality.
4. **Flatten Layer**: Flattens the output for the fully connected layers.
5. **Dense Layers**: Two dense layers with ReLU activation.
6. **Output Layer**: A dense layer with softmax activation for multi-class classification.

---

## **Training** 🏋️
The model is trained using the **Adam optimizer** and **categorical cross-entropy loss**. The training process includes:
- **Epochs**: 11
- **Batch Size**: 32
- **Validation Split**: 20% of the training data is used for validation.

---

## **Evaluation** 📈
The model is evaluated on the test dataset, and the following metrics are reported:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

---

## **Results** 🎯
The model achieves the following performance on the test dataset:
- **Accuracy**: 95.04%
- **Precision**: 95.00%
- **Recall**: 95.00%
- **F1-Score**: 95.00%

---
