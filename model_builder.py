"""
Prediction of age and gender using TensorFlow.
Dataset images: https://susanqq.github.io/UTKFace/
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D
from tensorflow.keras.models import load_model, Model, Sequential
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

num_epochs = 10

# For image resizing (default is 200x200)
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_DEPTH = 3

# Folders for image dataset and models
path_data = "dataset"
path_models = "models"

# Filenames for outputting the processed models
output_age_model = os.path.join(path_models, f'age_model_{num_epochs}epochs-v4.h5')
output_gender_model = os.path.join(path_models, f'gender_model_{num_epochs}epochs-v4.h5')

images = []
ages = []
genders = []

for img in os.listdir(path_data):
  age = img.split("_")[0]
  gender = img.split("_")[1]
  img = cv2.imread(os.path.join(path_data, img))

  if img.shape is not (IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH):
      resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_AREA)
      images.append(np.array(resized))
  else:
      images.append(np.array(img))

  ages.append(np.array(age))
  genders.append(np.array(gender))

ages = np.array(ages, np.int64)
images = np.array(images)
genders = np.array(genders, np.uint64)

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, ages, random_state=42)
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, genders, random_state=42)

############################################################

# Define age model and train
age_model = Sequential()

age_model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)))
age_model.add(MaxPool2D(pool_size=3, strides=2))
age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))
age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))
age_model.add(Flatten())
age_model.add(Dropout(0.2))
age_model.add(Dense(256, activation='relu'))
age_model.add(Dense(1, activation='linear', name='age'))

age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(age_model.summary())

history_age = age_model.fit(x_train_age, y_train_age, validation_data=(x_test_age, y_test_age), epochs=num_epochs)

age_model.save(output_age_model)

#Define gender model and train
gender_model = Sequential()

gender_model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)))
gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Conv2D(64, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Conv2D(128, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Conv2D(256, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Flatten())
gender_model.add(Dropout(0.2))
gender_model.add(Dense(256, activation='relu'))
gender_model.add(Dense(1, activation='sigmoid', name='gender'))

gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_gender = gender_model.fit(x_train_gender, y_train_gender, validation_data=(x_test_gender, y_test_gender), epochs=num_epochs)

gender_model.save(output_gender_model)

############################################################

history = history_age

# Plot the training and validation loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.title('Training and Validation Loss (Age)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

history = history_gender

# Plot the training and validation accuracy at each epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'y', label = 'training acc')
plt.plot(epochs, val_acc, 'r', label = 'validation acc')
plt.title('Training and Validation Accuracy (Gender)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

############################################################

# Test the model
my_model = load_model(output_gender_model, compile=False)
predictions = my_model.predict(x_test_gender)
y_pred = (predictions >= 0.5).astype(int)[:, 0]

print ("Accuracy =", metrics.accuracy_score(y_test_gender, y_pred))

# Verify accuracy of each class
cm = confusion_matrix(y_test_gender, y_pred)
sns.heatmap(cm, annot=True)
