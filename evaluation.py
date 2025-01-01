import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np


def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    # نمودار دقت
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # نمودار خطا
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()


def evaluate_model(model, test_generator, history):
    # ارزیابی مدل روی داده‌های تست
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # نمایش دقت آموزش
    train_accuracy = history.history['accuracy'][-1]
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

    # پیش‌بینی روی داده‌های تست
    Y_pred = np.argmax(model.predict(test_generator), axis=1)
    Y_true = test_generator.classes

    # رسم ماتریس درهم‌ریختگی
    conf_matrix = confusion_matrix(Y_true, Y_pred)
    ConfusionMatrixDisplay(conf_matrix, display_labels=list(
        test_generator.class_indices.keys())).plot()
    plt.title("Confusion Matrix")
    plt.show()

    # نمایش گزارش طبقه‌بندی
    print("Classification Report:")
    print(classification_report(Y_true, Y_pred,
          target_names=list(test_generator.class_indices.keys())))

    # رسم نمودار آموزش
    plot_training_history(history)

    # برگرداندن دقت مدل
    return test_accuracy
