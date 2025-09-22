import numpy as np
import deeplake
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K

# Import image dimensions from your data script
from data import IMG_HEIGHT, IMG_WIDTH

# --- Dice Coefficient (Must be identical to the one in train.py) ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# --- Function to load the ground truth masks for the test set ---
def load_ground_truth_test_masks():
    """
    Loads and processes the ground truth masks for the test set from Deeplake.
    Ensures INTER_NEAREST interpolation for sharp mask edges.
    """
    print("Loading ground-truth test masks from Deeplake...")
    try:
        ds_test = deeplake.load("hub://activeloop/glas-test", read_only=True)
        masks_list = [(mask.astype(np.uint8) * 255) for mask in ds_test.masks.numpy(aslist=True)]
        
        processed_masks = []
        for mask in masks_list:
            mask_resized = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
            processed_masks.append(mask_resized)
            
        # Convert to numpy array, add channel dim, and normalize to [0, 1] for metric calculation
        final_masks = np.array(processed_masks)[..., np.newaxis]
        return final_masks.astype('float32') / 255.

    except Exception as e:
        print(f"Failed to load test masks: {e}")
        return None

def visualize_sample(true_mask, prob_map, pred_binary):
    """Helper function to display a single comparison."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(np.squeeze(true_mask), cmap='gray')
    ax[0].set_title('Ground Truth Mask')
    ax[0].axis('off')

    ax[1].imshow(np.squeeze(prob_map), cmap='viridis') # Use a color map for probabilities
    ax[1].set_title('Predicted Probability Map')
    ax[1].axis('off')

    ax[2].imshow(np.squeeze(pred_binary), cmap='gray')
    ax[2].set_title('Final Prediction (Binarized)')
    ax[2].axis('off')

    plt.show()

def evaluate_model():
    # 1. Load the necessary data
    print("Loading ground truth masks and predicted probabilities...")
    true_masks = load_ground_truth_test_masks()
    predicted_probs = np.load('imgs_mask_test_predicted_probs.npy')

    if true_masks is None:
        print("Evaluation cannot proceed without ground truth masks.")
        return

    # 2. Find the optimal threshold by iterating through a range of values
    print("\nFinding the optimal threshold for binarization...")
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_score = 0
    best_threshold = 0

    true_masks_tf = tf.convert_to_tensor(true_masks, dtype=tf.float32)

    for threshold in thresholds:
        # Binarize the probability maps using the current threshold
        pred_binary_tf = tf.cast(predicted_probs > threshold, tf.float32)
        
        # Calculate the Dice score for this threshold
        score = dice_coef(true_masks_tf, pred_binary_tf).numpy()
        print(f"  Threshold: {threshold:.2f} -> Dice Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold

    # 3. Print the final, definitive result
    print("\n" + "="*40)
    print("        Final Model Evaluation Results        ")
    print("="*40)
    print(f"Optimal Threshold Found: {best_threshold:.2f}")
    print(f"Best Dice Coefficient on Test Set: {best_score:.4f}")
    print("="*40)

    # 4. Visualize a few random samples using the best threshold
    print("\nDisplaying a few visual samples with the optimal threshold...")
    num_samples_to_show = 5
    sample_indices = np.random.choice(len(true_masks), num_samples_to_show, replace=False)

    for i in sample_indices:
        print(f"\n--- Sample #{i} ---")
        pred_binary_sample = (predicted_probs[i] > best_threshold).astype(np.uint8)
        visualize_sample(true_masks[i], predicted_probs[i], pred_binary_sample)

if __name__ == '__main__':
    evaluate_model()