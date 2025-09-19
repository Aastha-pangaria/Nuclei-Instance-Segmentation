# import os
# import numpy as np
# import cv2
# import deeplake



# # --- Configuration ---
# # Modern deep learning models often work better with square images that are powers of 2.
# # We'll use 256x256 as a standard, but you can change this.
# IMG_HEIGHT = 256
# IMG_WIDTH = 256
# print(f"Target image size set to: {IMG_HEIGHT}x{IMG_WIDTH}")


# def process_and_save_data():
#     """
#     Loads the GlaS dataset from Deeplake, processes images and masks,
#     and saves them as .npy files.
#     """
#     # 1. Load Datasets from Activeloop Hub
#     # =======================================
#     print("-" * 30)
#     print("Loading Deeplake datasets...")
#     try:
#         ds_train = deeplake.load("hub://activeloop/glas-train", read_only=True)
#         ds_test = deeplake.load("hub://activeloop/glas-test", read_only=True)
#         print("Datasets loaded successfully.")
#     except Exception as e:
#         print(f"Failed to load datasets: {e}")
#         return

#     # 2. Process Training Data
#     # ========================
#     print("-" * 30)
#     print("Creating training images and masks...")
    
#     # Load images and masks into lists first, as they have different original sizes
#     train_images_list = ds_train.images.numpy(aslist=True)
#     train_masks_list = ds_train.masks.numpy(aslist=True)

#     processed_images_train = []
#     processed_masks_train = []

#     for img, mask in zip(train_images_list, train_masks_list):
#         # Convert image to grayscale to match your original script's logic
#         img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         img_resized = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
#         processed_images_train.append(img_resized)

#         # Ensure mask is grayscale and resize
#         mask_gray = np.squeeze(mask) # Remove any extra dimensions
#         mask_resized = cv2.resize(mask_gray, (IMG_WIDTH, IMG_HEIGHT))
        
#         # Binarize the mask: set all non-zero pixels to 255 (white)
#         mask_resized[mask_resized != 0] = 255
#         processed_masks_train.append(mask_resized)

#     print(f"Processed {len(processed_images_train)} training images and masks.")

#     # 3. Process Test Data
#     # ====================
#     print("-" * 30)
#     print("Creating test images...")
    
#     test_images_list = ds_test.images.numpy(aslist=True)
#     processed_images_test = []

#     for img in test_images_list:
#         img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         img_resized = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
#         processed_images_test.append(img_resized)

#     print(f"Processed {len(processed_images_test)} test images.")

#     # 4. Convert to NumPy arrays and save
#     # ===================================
#     print("-" * 30)
#     print("Converting to NumPy arrays and saving...")

#     # Convert lists to NumPy arrays
#     imgs_train_np = np.array(processed_images_train, dtype=np.uint8)
#     imgs_mask_train_np = np.array(processed_masks_train, dtype=np.uint8)
#     imgs_test_np = np.array(processed_images_test, dtype=np.uint8)

#     # Reshape to (N, H, W, 1) - Channels-Last format, standard for TensorFlow
#     # This is an improvement over the old (N, 1, H, W) format.
#     imgs_train_np = np.expand_dims(imgs_train_np, axis=-1)
#     imgs_mask_train_np = np.expand_dims(imgs_mask_train_np, axis=-1)
#     imgs_test_np = np.expand_dims(imgs_test_np, axis=-1)

#     np.save('imgs_train.npy', imgs_train_np)
#     np.save('imgs_mask_train.npy', imgs_mask_train_np)
#     np.save('imgs_test.npy', imgs_test_np)

#     print("Saving to .npy files done.")
#     print(f"Train images shape: {imgs_train_np.shape}")
#     print(f"Train masks shape:  {imgs_mask_train_np.shape}")
#     print(f"Test images shape:  {imgs_test_np.shape}")


# def load_train_data():
#     """Loads the processed training data from .npy files."""
#     imgs_train = np.load('imgs_train.npy')
#     imgs_mask_train = np.load('imgs_mask_train.npy')
#     return imgs_train, imgs_mask_train


# def load_test_data():
#     """Loads the processed test data from a .npy file."""
#     imgs_test = np.load('imgs_test.npy')
#     return imgs_test


# if __name__ == '__main__':
#     # This will run the entire process of downloading, processing, and saving the data.
#     process_and_save_data()

#     # You can then use the load functions in your training script like before.
#     print("\nExample of loading the saved data:")
#     X_train, y_train = load_train_data()
#     print(f"Loaded training images with shape: {X_train.shape}")


import os
import numpy as np
import cv2
import deeplake

# --- Configuration ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
print(f"Target image size set to: {IMG_HEIGHT}x{IMG_WIDTH}")


def process_and_save_data():
    """
    Loads the GlaS dataset from Deeplake, processes images, masks, and IDs,
    and saves them as .npy files.
    """
    # 1. Load Datasets from Activeloop Hub
    # =======================================
    print("-" * 30)
    print("Loading Deeplake datasets...")
    try:
        ds_train = deeplake.load("hub://activeloop/glas-train", read_only=True)
        ds_test = deeplake.load("hub://activeloop/glas-test", read_only=True)
        print("Datasets loaded successfully.")
    except Exception as e:
        print(f"Failed to load datasets: {e}")
        return

    # 2. Process Training Data
    # ========================
    print("-" * 30)
    print("Creating training images and masks...")
    
    train_images_list = ds_train.images.numpy(aslist=True)
    # The segmentation masks are under the 'masks' tensor
    train_masks_list = ds_train.masks.numpy(aslist=True)

    processed_images_train = []
    processed_masks_train = []

    for img, mask in zip(train_images_list, train_masks_list):
        # Process image
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
        processed_images_train.append(img_resized)

        # Process mask
        mask_gray = np.squeeze(mask)
        # --- FIX: Convert boolean mask to uint8 before resizing ---
        mask_uint8 = mask_gray.astype(np.uint8) 
        mask_resized = cv2.resize(mask_uint8, (IMG_WIDTH, IMG_HEIGHT))
        
        # Binarize the mask: set all non-zero pixels to 255 (white)
        mask_resized[mask_resized != 0] = 255
        processed_masks_train.append(mask_resized)

    print(f"Processed {len(processed_images_train)} training images and masks.")

    # 3. Process Test Data and IDs
    # ============================
    print("-" * 30)
    print("Creating test images and IDs...")
    
    test_images_list = ds_test.images.numpy(aslist=True)
    processed_images_test = []
    processed_ids_test = []

    # Using a simple counter for test IDs since none are provided in the dataset
    for i, img in enumerate(test_images_list):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_gray, (IMG_WIDTH, IMG_HEIGHT))
        processed_images_test.append(img_resized)
        processed_ids_test.append(f"test_{i+1}") # Create a string ID

    print(f"Processed {len(processed_images_test)} test images.")

    # 4. Convert to NumPy arrays and save
    # ===================================
    print("-" * 30)
    print("Converting to NumPy arrays and saving...")

    # Ensure final arrays have the channel dimension for TensorFlow
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
