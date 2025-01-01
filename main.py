from evaluation import evaluate_model
from train import train_model
from model import create_model, load_existing_model
from preprocess import get_data_generators
import numpy as np
import tensorflow as tf
import random

# تنظیم seed ثابت برای تکرارپذیری نتایج
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# مسیر پوشه‌های دیتاست
train_dir = "data/Training"
test_dir = "data/Testing"

# بارگذاری داده‌ها از دیتاست
train_generator, val_generator, test_generator = get_data_generators(
    train_dir, test_dir)

# بارگذاری مدل ذخیره شده یا ساخت مدل جدید
model = load_existing_model("brain_tumor_model.keras") or create_model()

# آموزش مدل
history = train_model(model, train_generator, val_generator)

# ذخیره مدل پس از آموزش
model.save("brain_tumor_model.keras")

# ارزیابی مدل
evaluate_model(model, test_generator, history)
