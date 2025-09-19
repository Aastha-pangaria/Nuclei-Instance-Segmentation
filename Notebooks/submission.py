import numpy as np
import cv2
from itertools import chain

# Import the configuration and data loading function from our processing script
from data import IMG_HEIGHT, IMG_WIDTH, load_test_data

def prep(img):
    """
    Prepares a predicted mask for RLE.
    It thresholds the mask and resizes it to the correct dimensions.
    """
    # The model output is likely float32 probabilities (0.0 to 1.0)
    img = img.astype('float32')
    
    # Threshold at 0.5 to create a binary mask
    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    
    # Resize to the final submission dimensions (must match original if specified by a competition)
    # Here we use the same dimensions we trained on.
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img


def run_length_enc(label):
    """
    Converts a binary mask into a run-length encoded string.
    """
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 1:  # If the mask is empty, return an empty string
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z + 1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s + 1, l + 1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def create_submission_file():
    """
    Loads predicted masks and their IDs, generates the RLE for each,
    and saves the result to submission.csv.
    """
    # --- IMPORTANT ---
    # This script assumes you have already trained a model and used it
    # to predict masks on the test set. Those predictions must be saved
    # to a file named 'imgs_mask_test_predicted.npy'.
    PREDICTED_MASKS_FILE = 'imgs_mask_test_predicted.npy'
    
    try:
        predicted_masks = np.load(PREDICTED_MASKS_FILE)
    except FileNotFoundError:
        print(f"ERROR: Predicted masks file not found at '{PREDICTED_MASKS_FILE}'")
        print("Please train your model, run predictions on 'imgs_test.npy', and save the results first.")
        return

    # Load the corresponding test IDs. We don't need the actual test images here.
    _, imgs_id_test = load_test_data()

    print(f"Loaded {len(predicted_masks)} predicted masks.")
    
    # No need to sort if IDs were generated sequentially
    total = predicted_masks.shape[0]
    ids = []
    rles = []

    print("Generating Run-Length Encodings...")
    for i in range(total):
        # The predicted mask shape is likely (H, W, 1)
        img = predicted_masks[i, :, :, 0]
        
        # Prepare the mask (threshold, resize)
        img_prepped = prep(img)
        
        # Encode it
        rle = run_length_enc(img_prepped)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if (i + 1) % 50 == 0:
            print('Done: {0}/{1} images'.format(i + 1, total))

    # Write the submission.csv file
    file_name = 'submission.csv'
    with open(file_name, 'w+') as f:
        f.write('img,pixels\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')
    
    print(f"\nSubmission file '{file_name}' created successfully.")


if __name__ == '__main__':
    create_submission_file()
