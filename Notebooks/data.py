import os
import numpy as np
import cv2
import deeplake

IMG_HEIGHT = 256
IMG_WIDTH = 256
print(f"Target image size set to: {IMG_HEIGHT}x{IMG_WIDTH}")


def process_and_save_data():
    """
    Loads the GlaS dataset from Deeplake, processes images, masks, and IDs,
    and saves them as .npy files.
    """
    print("-" * 30)
    print("Loading Deeplake datasets...")
    try:
        ds_train = deeplake.load("hub://activeloop/glas-train", read_only=True)
        ds_test = deeplake.load("hub://activeloop/glas-test", read_only=True)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return

    print("-" * 30)
    print("Creating training images and masks...")
    
    train_images_list = ds_train.images.numpy(aslist=True)
    train_masks_list = ds_train.masks.numpy(aslist=True)

    processed_images_train = []
    processed_masks_train = []

    for img, mask in zip(train_images_list, train_masks_list):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
        processed_images_train.append(img_resized)

        mask_gray = np.squeeze(mask)
        mask_uint8 = mask_gray.astype(np.uint8) 
        mask_resized = cv2.resize(mask_uint8, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        mask_resized[mask_resized != 0] = 255
        processed_masks_train.append(mask_resized)

    print(f"Processed {len(processed_images_train)} training images and masks.")

    print("-" * 30)
    print("Creating test images and IDs...")
    
    test_images_list = ds_test.images.numpy(aslist=True)
    processed_images_test = []
    processed_ids_test = []

    for i, img in enumerate(test_images_list):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
        processed_images_test.append(img_resized)
        processed_ids_test.append(f"test_{i+1}") 

    print(f"Processed {len(processed_images_test)} test images.")

    print("-" * 30)
    print("Converting to NumPy arrays and saving...")

    imgs_train_np = np.array(processed_images_train, dtype=np.uint8)[..., np.newaxis]
    imgs_mask_train_np = np.array(processed_masks_train, dtype=np.uint8)[..., np.newaxis]
    imgs_test_np = np.array(processed_images_test, dtype=np.uint8)[..., np.newaxis]
    ids_test_np = np.array(processed_ids_test)

    np.save('imgs_train.npy', imgs_train_np)
    np.save('imgs_mask_train.npy', imgs_mask_train_np)
    np.save('imgs_test.npy', imgs_test_np)
    np.save('imgs_id_test.npy', ids_test_np)

    print("Saving to .npy files done.")
    print(f"Train images shape: {imgs_train_np.shape}")
    print(f"Train masks shape:  {imgs_mask_train_np.shape}")
    print(f"Test images shape:  {imgs_test_np.shape}")
    print(f"Test IDs shape:     {ids_test_np.shape}")


def load_train_data():
    """Loads the processed training data from .npy files."""
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def load_test_data():
    """Loads the processed test data and IDs from .npy files."""
    imgs_test = np.load('imgs_test.npy')
    imgs_id_test = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id_test


if __name__ == '__main__':
    process_and_save_data()
