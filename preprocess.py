from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128
BATCH_SIZE = 32


def get_data_generators(train_dir, test_dir):
    # تولید داده‌های افزایشی برای آموزش
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2  # 20 درصد برای اعتبارسنجی
    )

    # پیش‌پردازش داده‌های تست
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # داده‌های آموزشی
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    # داده‌های اعتبارسنجی
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=True,
        seed=42,
    )

    # داده‌های تست
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator
