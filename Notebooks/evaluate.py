import numpy as np
import deeplake
import cv2
from tensorflow.keras import backend as K

# Import the image dimensions from your data script
from data import IMG_HEIGHT, IMG_WIDTH

# --- Dice Coefficient Function (from train.py) ---
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# --- Function to load and process the ground truth masks for the test set ---
def load_ground_truth_test_masks():
    print("Loading ground-truth test masks from Deeplake...")
    try:
        ds_test = deeplake.load("hub://activeloop/glas-test", read_only=True)
        # The masks are boolean, so convert to uint8 (0s and 255s)
        masks_list = [(mask.astype(np.uint8) * 255) for mask in ds_test.masks.numpy(aslist=True)]
        
        processed_masks = []
        for mask in masks_list:
            # Resize and add channel dimension, same as in data.py
            mask_resized = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
            processed_masks.append(mask_resized)
            
        # Convert to numpy array and normalize to [0, 1]
        final_masks = np.array(processed_masks).astype('float32') / 255.
        return final_masks[..., np.newaxis] # Add channel dimension

    except Exception as e:
        print(f"Failed to load test masks: {e}")
        return None

def evaluate_model():
    # Load your model's predictions
    print("Loading model predictions...")
    predicted_masks = np.load('imgs_mask_test_predicted.npy')

    # Load the actual correct masks
    true_masks = load_ground_truth_test_masks()

    if true_masks is None:
        print("Could not proceed with evaluation.")
        return

    print(f"\nComparing {len(predicted_masks)} predicted masks against ground truth...")
    
    # Binarize the predictions (convert probabilities to 0 or 1)
    predicted_masks[predicted_masks > 0.5] = 1
    predicted_masks[predicted_masks <= 0.5] = 0

    # Calculate the Dice Coefficient using TensorFlow backend
    final_score = dice_coef(true_masks, predicted_masks)
    
    print("\n" + "="*30)
    print(f"Final Model Performance on Test Set:")
    print(f"  Mean Dice Coefficient: {final_score.numpy():.4f}")
    print("="*30)

if __name__ == '__main__':
    evaluate_model()
