import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Import the prep function from our modernized submissions script
from submission import prep
# Import the correct image dimensions from our data script
from data import IMG_HEIGHT, IMG_WIDTH


def visualize_results():
    """
    Loads test images and their predicted masks, and displays
    a random sample with the mask overlaid on the original image.
    """
    print("Loading test images and predicted masks...")
    images = np.load('imgs_test.npy')
    # --- FIX: Load the PREDICTED masks, not the ground truth ---
    predicted_masks = np.load('imgs_mask_test_predicted.npy')
    
    total_images = images.shape[0]
    
    # --- Improvement: Visualize 5 random images instead of all of them ---
    num_to_visualize = 5
    image_indices = random.sample(range(total_images), num_to_visualize)
    
    print(f"Displaying {num_to_visualize} random results...")
    for i in image_indices:
        # --- FIX: Use np.squeeze to handle the TensorFlow channel dimension ---
        image = np.squeeze(images[i])
        result = np.squeeze(predicted_masks[i])
        
        # Process the raw prediction mask the same way as for submission
        result = prep(result)
        
        # Create a figure to display the images
        plt.figure(figsize=(8, 8))
        plt.title(f"Image and Predicted Mask #{i+1}")
        
        # Display the original grayscale image
        plt.imshow(image, cmap='gray')
        
        # Overlay the predicted mask with transparency
        # Use a masked array to only show the "on" pixels of the mask
        masked_result = np.ma.masked_where(result == 0, result)
        plt.imshow(masked_result, cmap='viridis', alpha=0.5) # Use a bright color for the mask
        
        plt.show()


if __name__ == '__main__':
    visualize_results()

