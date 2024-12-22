"""
Alzheimer Detection and Classification using Convolutional Neural Networks.

Based on Naman Kumar's Implementation with some modifications.

Kaggle Notebook:
https://www.kaggle.com/code/namank24/alzheimer-detection-and-classification

Research Paper: 
https://www.afjbs.com/uploads/paper/77773b106f02301713fa3a488ee52e3b.pdf
"""

# Import Libraries
import tensorflow as tf
import splitfolders
import keras
import plotext as plt
import time


# Constants
DATA_DIR = 'Dataset' # Dataset
WORK_DIR = 'DatasetOut' # Output after splitting dataset
IMG_HEIGHT = 128 # Input height 
IMG_WIDTH = 128 # Input width
EPOCHS = 100 # training epochs
BATCH_SIZE = 64 # Batch Size

# Split Dataset
# 80% Training, 10% Validation, 10% Testing
print("[INFO] Splitting Dataset")
splitfolders.ratio(DATA_DIR, output=WORK_DIR, seed=1345, ratio=(.8, 0.1,0.1))

# Load Dataset

# Training Set
print("[INFO] Training Set:")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(f"{WORK_DIR}/train",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64
    )

print("[INFO] Test Set:")
test_ds = tf.keras.preprocessing.image_dataset_from_directory(f"{WORK_DIR}/test",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64
    )

print("[INFO] Validation Set:")
val_ds = tf.keras.preprocessing.image_dataset_from_directory(f"{WORK_DIR}/val",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=64
    )

print("[INFO] Class Names:")
print(f"- {name}" for name in train_ds.class_names)

# Model Creation
model = keras.models.Sequential()

model.add(keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(keras.layers.Rescaling(1./255))
model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.20))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer="he_normal"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(64, activation="relu"))
model.add(keras.layers.Dense(4, activation="softmax"))

# Model Compilation
model.compile(loss="sparse_categorical_crossentropy", optimizer = "Adam",metrics=["accuracy"])

print("[INFO] Model Summary:")
model.summary()

# Train Model
print("[INFO] Training Model:")
start_time = time.time()
hist = model.fit(train_ds,
                 validation_data=val_ds,
                 epochs=EPOCHS, 
                 batch_size=BATCH_SIZE, 
                 verbose=2)
end_time = time.time()

print(f"[INFO] Training Time: {round(end_time-start_time, 2)}s")

train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']


epochs = range(len(train_acc))
plt.clear_figure()
plt.plot(epochs, train_acc)
plt.title("Training Accuracy")
plt.show()

plt.clear_figure()
plt.plot(epochs, train_loss)
plt.title("Training Loss")
plt.show()

plt.clear_figure()
plt.plot(epochs, val_acc)
plt.title("Validation Accuracy")
plt.show()

plt.clear_figure()
plt.plot(epochs, val_loss)
plt.title("Validation Loss")
plt.show()

# Save Model
print("[INFO] Saving Model to model.keras:")
model.save("model.keras",overwrite=True)

# Evaluate Model
print("[INFO] Evaluating Model:")
loss, accuracy = model.evaluate(test_ds)

print(f"Loss: {round(loss, 4)}")
print(f"Accuracy: {round(accuracy*100, 2)}%")
