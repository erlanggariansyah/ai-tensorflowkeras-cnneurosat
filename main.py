import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

dataset_path = os.path.join(os.getcwd(), 'dataset')

target_size = (24, 24)
batch_size = 100
datagen = ImageDataGenerator(rescale=1./255)
class_names = sorted(os.listdir(dataset_path))

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=target_size,
    batch_size=batch_size,
    classes=class_names,
    shuffle=True
)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(class_names), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

result = model.fit(train_data, epochs=10)

# Grafik loss
train_loss = result.history['loss']
plt.plot(range(1, len(train_loss) + 1), train_loss, 'r', label='Training Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Grafik akurasi
train_accuracy = result.history['accuracy']
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'b', label='Training Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
