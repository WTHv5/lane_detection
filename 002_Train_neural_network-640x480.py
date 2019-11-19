
# coding: utf-8

# In[69]:

import keras as K
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GaussianNoise, Cropping2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.engine.input_layer import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard

from keras import backend as K


# In[36]:

#DATASET_CSV_PATH='bdd_train_set.csv'
DATASET_CSV_PATH='bdd_and_ber_train_set.csv'
MAX_NUMBER_OF_LANES=4

samples=pd.read_csv(DATASET_CSV_PATH)
samples=samples[samples['lane']>=0]
samples=samples[samples['lane']<MAX_NUMBER_OF_LANES]
samples=samples[samples['nr_lanes']>=0]
samples=samples[samples['nr_lanes']<MAX_NUMBER_OF_LANES]
nr_train_samples=len(samples)
samples.hist()



# In[19]:

samples['lane'].value_counts()


# In[20]:




# In[52]:

train_samples=samples.groupby('lane').apply(pd.DataFrame.sample, n=len(samples),replace=True).reset_index(drop=True).sample(frac=1, replace=False)
train_samples['lane']=train_samples['lane'].astype('str')
#train_samples.head()
train_samples['lane'].value_counts()


# In[53]:

train_samples.head()


# In[77]:

DATASET_CSV_PATH='val_data.csv'
MAX_NUMBER_OF_LANES=4

val_samples=pd.read_csv(DATASET_CSV_PATH)
val_samples=val_samples[val_samples['lane']>=0]
val_samples=val_samples[val_samples['lane']<MAX_NUMBER_OF_LANES]
val_samples=val_samples[val_samples['nr_lanes']>=0]
val_samples=val_samples[val_samples['nr_lanes']<MAX_NUMBER_OF_LANES]
val_samples['lane']=val_samples['lane'].astype('str')
nr_val_samples=len(val_samples)
#val_samples['lane'].hist()


# In[78]:

val_samples.head()


# In[79]:

val_samples['lane'].value_counts()


# In[82]:

# dimensions of our images.
img_width, img_height = 640, 480
CROP_TOP=int(200)
CROP_BOTTOM=int(30)


nb_train_samples = nr_train_samples
nb_validation_samples = nr_val_samples
batch_size = 8

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_height, img_width)
else:
    input_shape = (img_height, img_width, 3)
    
model = Sequential()
model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0, 0)), input_shape=input_shape))
model.add(GaussianNoise(0.5))

#model.add(GaussianNoise(0.5, input_shape=input_shape))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))


# In[86]:

#model=darknet()
sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.00001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


# In[87]:

model.summary()


# In[ ]:




# In[96]:

MODEL_SAVE_PATH='models/weights.{epoch:02d}-{val_loss:.2f}-{acc:.2f}.hdf5'

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    shear_range=0.4,
    zoom_range=0.1,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.1,
    horizontal_flip=False,
    brightness_range=(0.25,1.85),
    channel_shift_range=130)
#)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

x_col='img_path'
y_col='lane'
train_generator = train_datagen.flow_from_dataframe(
    train_samples,
    x_col=x_col,
    y_col=y_col,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_dataframe(
    val_samples,
    x_col=x_col,
    y_col=y_col,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

callbacks = [#EarlyStopping(monitor='val_loss', patience=10),
             CSVLogger('./train.log', separator=',', append=False),
             #TensorBoard(log_dir='./tensorboard', histogram_freq=100, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch'),
             ModelCheckpoint(filepath=MODEL_SAVE_PATH, monitor='acc', verbose=True, save_best_only=False, mode='max')]


# In[ ]:




# In[93]:

epochs = 1000
iters_per_epoch=200
model.fit_generator(
    train_generator,
    callbacks=callbacks,
    steps_per_epoch=iters_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=200,
    max_queue_size=64,
    workers=24,
    use_multiprocessing=True)

