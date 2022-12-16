#%% Import library
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.utils as utils
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator

from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
tf.config.list_physical_devices()

import warnings
warnings.filterwarnings("ignore")


#%% loading data
dogs = glob.glob('Train_Data/dog/*.jpg')
dog_labels = ['dog'] * len(dogs)
cats = glob.glob('Train_Data/cat/*/jpg')
cat_labels = ['cat'] * len(cats)

labels = dog_labels + cat_labels
image_links = dogs + cats

# Use stratify for balance between labels in train and test data
images_train, images_val, y_label_train, y_label_val = \
    train_test_split(image_links, labels, test_size=0.2, stratify=labels)

# generate data (data augmentation)
dict_label = {'dog': 0, 'cat': 1}
train_generator = DataGenerator(images_train, y_label_train, batch_size=32, index2class=dict_label,
                                input_dim=(224, 224), n_channels=3, n_classes=2, normalize=False)
val_generator = DataGenerator(images_val, y_label_val, batch_size=32, index2class=dict_label,
                              input_dim=(224, 224), n_channels=3, n_classes=2, normalize=False)

#%% Check DataGenerator
check_aug = ['Train_Data/cat/cat.100.jpg'] * 32
check_generator = DataGenerator(check_aug, y_label_train, batch_size=32, index2class=dict_label,
                                input_dim=(224, 224), n_channels=3, n_classes=2, normalize=False)

X_batch, y_batch = check_generator.__getitem__(0)
print(X_batch.shape, y_batch.shape)

fg, ax = plt.subplots(4, 5, figsize=(20, 16))
fg.suptitle('Augmentation Images')

for i in np.arange(4):
    for j in np.arange(5):
        ax[i, j].imshow(X_batch[i + j + j * i] / 255.0)
        ax[i, j].set_xlabel('Image ' + str(i + j + j * i))
        ax[i, j].axis('off')
plt.show()

#%% Training transfer learning model
base_network = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
flat = Flatten()
dense = Dense(1, activation='sigmoid')

model = Sequential([base_network, flat, dense])
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

#%% Frozen base_network
for layer in model.layers[: 1]:
    layer.trainable = False

for layer in model.layers:
    print(f'Layer {layer} trainable {layer.trainable}')

# %% Train model for 1 epoch
with tf.device('/device:GPU:0'):
    model.fit(train_generator, steps_per_epoch=len(train_generator), \
        validation_data = val_generator, validation_steps=5, epochs=10)
    
# %%
