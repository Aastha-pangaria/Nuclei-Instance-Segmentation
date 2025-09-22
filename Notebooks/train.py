import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data import load_train_data, load_test_data, IMG_HEIGHT, IMG_WIDTH

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, UpSampling2D, Concatenate, BatchNormalization, LeakyReLU)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2 # Import the pre-trained model
from tensorflow.keras import backend as K

# --- Loss Function (Using the stable Hybrid Loss is better for a new start) ---
smooth = 1e-6
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))

# --- NEW: Model with Pre-trained MobileNetV2 Encoder ---
def create_unet_with_transfer_learning():
    # Input layer must have 3 channels for pre-trained models
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # 1. Load the pre-trained MobileNetV2 as the encoder
    encoder = MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
    
    # We don't want to re-train the early layers
    encoder.trainable = False

    # 2. Get the skip connection outputs from the encoder
    skip_connection_names = [
        'block_1_expand_relu',  # 128x128
        'block_3_expand_relu',  # 64x64
        'block_6_expand_relu',  # 32x32
        'block_13_expand_relu', # 16x16
    ]
    encoder_outputs = [encoder.get_layer(name).output for name in skip_connection_names]
    
    # 3. Define the bottleneck
    bottleneck = encoder.get_layer('block_16_project').output # 8x8

    # 4. Define the decoder (the part we will train)
    def decoder_block(input_tensor, skip_tensor, num_filters):
        x = UpSampling2D((2, 2))(input_tensor)
        x = Concatenate()([x, skip_tensor])
        x = Conv2D(num_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(num_filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    d1 = decoder_block(bottleneck, encoder_outputs[3], 512)
    d2 = decoder_block(d1, encoder_outputs[2], 256)
    d3 = decoder_block(d2, encoder_outputs[1], 128)
    d4 = decoder_block(d3, encoder_outputs[0], 64)

    # Final output layer
    final_upsample = UpSampling2D((2,2))(d4)
    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(final_upsample)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_and_predict():
    # Load data
    print("Loading train data...")
    imgs_train, imgs_mask = load_train_data() # uint8, shape (N, H, W, 1)

    # --- CHANGE: Convert grayscale to 3-channel for the pre-trained encoder ---
    imgs_train_rgb = np.repeat(imgs_train, 3, axis=-1)
    
    # Normalize images and masks
    imgs_train_rgb = imgs_train_rgb.astype('float32') / 255.0
    imgs_mask = imgs_mask.astype('float32') / 255.0

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        imgs_train_rgb, imgs_mask, test_size=0.15, random_state=42
    )

    # Model
    print("Creating U-Net model with pre-trained MobileNetV2 encoder...")
    model = create_unet_with_transfer_learning()
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=bce_dice_loss,
        metrics=[dice_coef]
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-6),
        ModelCheckpoint('mobilenet_unet.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Training
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Prediction
    print("Loading and preprocessing test data...")
    imgs_test, _ = load_test_data()
    imgs_test_rgb = np.repeat(imgs_test, 3, axis=-1)
    imgs_test_rgb = imgs_test_rgb.astype('float32') / 255.0
    
    print("Predicting on test set...")
    predicted_probs = model.predict(imgs_test_rgb, verbose=1)
    np.save('imgs_mask_test_predicted_probs.npy', predicted_probs)
    print("Predictions saved.")

if __name__ == '__main__':
    train_and_predict()