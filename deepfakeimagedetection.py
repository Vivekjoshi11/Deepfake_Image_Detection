#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from PIL import Image

# Define the paths to your image folders
fake_folder =r"C:\Users\vj785\OneDrive\Desktop\8thsem project\imaged\train1\fake2"
real_folder = r"C:\Users\vj785\OneDrive\Desktop\8thsem project\imaged\train1\real2"

# Function to load and preprocess images from a folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            img = img.resize((224, 224))  # Resize the image to a common size
            img = np.array(img) / 255.0   # Normalize pixel values to [0, 1]
            images.append(img)
    return np.array(images)

# Load and preprocess fake images
fake_images = load_images_from_folder(fake_folder)
fake_labels = np.zeros(len(fake_images))  # Label fake images as 0

# Load and preprocess real images
real_images = load_images_from_folder(real_folder)
real_labels = np.ones(len(real_images))   # Label real images as 1

# Combine fake and real images into a single dataset
X = np.concatenate((fake_images, real_images), axis=0)
y = np.concatenate((fake_labels, real_labels), axis=0)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)


# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()


# In[3]:


history = model.fit(X, y, epochs=15, batch_size=32, validation_split=0.2)


# In[4]:


test_loss, test_acc = model.evaluate(X, y)
print('Test accuracy:', test_acc)


# In[5]:


def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to match model input size
    img = np.array(img) / 255.0   # Normalize pixel values to [0, 1]
    return img

# Define a function to predict whether an image is real or fake
def predict_image_real_or_fake(image_path, model):
    img = load_and_preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    if prediction[0][0] >= 0.5:
        return "Real"
    else:
        return "Fake"

# Test the model on a sample image
test_image_path = r"D:\kinnari\vivekprofiledpw.jpg" # Replace with the path to your test image
prediction = predict_image_real_or_fake(test_image_path, model)
print("Prediction for", test_image_path, ":", prediction)


# In[6]:


model.save('fake_image_detection_model.keras')


# In[7]:


for layer in reversed(model.layers):
    if 'conv' in layer.name:
        last_conv_layer_name = layer.name
        break

print("Name of last convolutional layer:", last_conv_layer_name)


# In[11]:


from PIL import Image
import numpy as np
from IPython.display import display

# Assuming the rest of your code is already defined...

test_image_path =r"D:\kinnari\vivekprofiledpw.jpg" # Replace with the path to your test image
prediction = predict_image_real_or_fake(test_image_path, model)
print("Prediction for", test_image_path, ":\n\033[1m", prediction.upper(), "\033[0m")
# Show the image
img = Image.open(test_image_path)
display(img)


# In[ ]:




