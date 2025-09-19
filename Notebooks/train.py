import numpy as np
import cv2

# Import our new data loading functions and configuration
from data import load_train_data, load_test_data, IMG_HEIGHT, IMG_WIDTH

# --- Corrected Keras / TensorFlow Imports ---
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, LeakyReLU, SpatialDropout2D, 
                                     AveragePooling2D, UpSampling2D, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K

# --- Dice Coefficient Metric/Loss ---
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


# --- Modernized U-Net Model Architecture ---
def create_model():
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # Encoder
    conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = SpatialDropout2D(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = SpatialDropout2D(0.2)(conv3)

    # Decoder
    up4 = UpSampling2D(size=(2, 2))(conv3)
    merge4 = Concatenate(axis=-1)([conv2, up4])
    conv4 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(merge4)
    conv4 = LeakyReLU()(conv4)
    conv4 = SpatialDropout2D(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = SpatialDropout2D(0.2)(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    merge5 = Concatenate(axis=-1)([conv1, up5])
    conv5 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(merge5)
    conv5 = LeakyReLU()(conv5)
    conv5 = SpatialDropout2D(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = LeakyReLU()(conv5)
    conv5 = SpatialDropout2D(0.2)(conv5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=3e-4), loss=dice_coef_loss, metrics=[dice_coef])
    return model


# --- Main Training and Prediction Function ---
def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = imgs_train.astype('float32')
    # Calculate mean and std ONLY from training data
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32') / 255.

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = create_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, verbose=1),
        ModelCheckpoint('weights.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,
        shear_range=0.15, horizontal_flip=True, vertical_flip=True, fill_mode='reflect'
    )
    
    print('-'*30)
    print('Begin training...')
    print('-'*30)
    history = model.fit(
        datagen.flow(imgs_train, imgs_mask_train, batch_size=4),
        epochs=100, verbose=1, shuffle=True,
        validation_data=(imgs_train, imgs_mask_train),
        callbacks=callbacks, steps_per_epoch=len(imgs_train) // 4
    )

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, _ = load_test_data()
    imgs_test = imgs_test.astype('float32')
    # Use the SAME mean and std from training data to normalize test data
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.keras')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    predicted_masks = model.predict(imgs_test, verbose=1)
    
    np.save('imgs_mask_test_predicted.npy', predicted_masks)
    print("Predictions saved to 'imgs_mask_test_predicted.npy'")


if __name__ == '__main__':
    train_and_predict()

