# # 🖼️ CIFAR-10 Image Classification using TensorFlow
# Complete deep learning pipeline with training, evaluation, and visualizations.


# ## 📦 Step 1: Import Libraries and Load CIFAR-10 Dataset

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Show shapes
print("Training set:", x_train.shape, y_train.shape)
print("Test set:", x_test.shape, y_test.shape)


# ## ⚙️ Step 2: Normalize Images and One-Hot Encode Labels


x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

from tensorflow.keras.utils import to_categorical
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print("x_train:", x_train.shape, "y_train_cat:", y_train_cat.shape)

# ## 🧠 Step 3: Build the CNN Model


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()

# ## 🏃 Step 4: Compile and Train the Model


from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train, y_train_cat,
    epochs=15,
    batch_size=64,
    validation_data=(x_test, y_test_cat)
)


# ## 📊 Step 5: Visualize Accuracy and Loss


plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ## 🧪 Step 6: Evaluate Model and Show Sample Predictions

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"✅ Test Accuracy: {test_accuracy:.4f}")
print(f"❌ Test Loss: {test_loss:.4f}")

# Make predictions
predictions = model.predict(x_test)

# Show sample predictions
plt.figure(figsize=(10, 5))
plt.suptitle(" Final Results: Model Predictions vs Actual Labels", fontsize=16)
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[i])
    pred = class_names[np.argmax(predictions[i])]
    actual = class_names[np.argmax(y_test_cat[i])]
    plt.title(f"Pred: {pred}\nTrue: {actual}", fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()