import os
import json
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

# ── Đường dẫn ────────────────────────────────────────────────────────────────
BASE_DIR    = r"D:\Computer Vision Project"
DATA_DIR    = os.path.join(BASE_DIR, "data", "Character dataset")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Tham số huấn luyện ────────────────────────────────────────────────────────
IMG_SIZE   = (28, 28)
BATCH_SIZE = 30
EPOCHS     = 100

# ── Timestamp cho tên file log & checkpoint ───────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH        = os.path.join(MODELS_DIR, f"train_log_{timestamp}.txt")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_char_model.keras")

# ── Data generators (80% train / 20% val) ────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

num_classes = train_generator.num_classes
print(f"[INFO] Số lớp ký tự: {num_classes}")
print(f"[INFO] Train samples : {train_generator.samples}")
print(f"[INFO] Val   samples : {val_generator.samples}")

# ── Kiến trúc CNN ─────────────────────────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same",
           input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Callback ghi log ra file .txt ─────────────────────────────────────────────
class TxtLogger(Callback):
    def __init__(self, log_path: str):
        super().__init__()
        self.log_path = log_path

    def on_train_begin(self, logs=None):
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"Training started at {datetime.datetime.now()}\n")
            f.write(f"Dataset   : {DATA_DIR}\n")
            f.write(f"Epochs    : {EPOCHS}\n")
            f.write(f"Batch size: {BATCH_SIZE}\n")
            f.write(f"Num classes: {num_classes}\n")
            f.write("-" * 60 + "\n")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        line = (
            f"Epoch {epoch + 1:>3}/{EPOCHS} | "
            f"loss: {logs.get('loss', 0):.4f} | "
            f"acc: {logs.get('accuracy', 0):.4f} | "
            f"val_loss: {logs.get('val_loss', 0):.4f} | "
            f"val_acc: {logs.get('val_accuracy', 0):.4f}\n"
        )
        print(line, end="")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def on_train_end(self, logs=None):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("-" * 60 + "\n")
            f.write(f"Training finished at {datetime.datetime.now()}\n")

# ── Callback lưu best checkpoint ─────────────────────────────────────────────
checkpoint_cb = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

# ── Huấn luyện ───────────────────────────────────────────────────────────────
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, TxtLogger(LOG_PATH)],
)

# ── Lưu nhãn ─────────────────────────────────────────────────────────────────
labels_dict = {v: k for k, v in train_generator.class_indices.items()}
labels_path = os.path.join(MODELS_DIR, "labels_dict.json")
with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(labels_dict, f, ensure_ascii=False, indent=2)

print(f"\n[DONE] Best checkpoint : {CHECKPOINT_PATH}")
print(f"[DONE] Training log    : {LOG_PATH}")
print(f"[DONE] Labels dict     : {labels_path}")