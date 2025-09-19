# Gland Segmentation in Colon Histology Images using U-Net
This project implements a complete deep learning pipeline to perform instance segmentation on high-resolution digital pathology images. A U-Net model is trained to identify and segment glandular structures in colorectal cancer histology slides from the Warwick-QU GlaS challenge dataset.

## Project Overview
The goal is to automate the process of gland segmentation in histology images, a crucial but time-consuming task for pathologists. Accurate segmentation is vital for cancer grading and diagnosis. This project leverages a U-Net architecture, known for its effectiveness in biomedical image segmentation, and trains it on the public GlaS challenge dataset streamed directly from the cloud using Deeplake.

The repository contains a full, end-to-end workflow:

Data Processing: Fetches data, resizes images and masks, and saves them locally.

Model Training: Builds and trains a U-Net model with data augmentation.

Evaluation: Measures the model's performance on the unseen test set.

Prediction & Submission: Generates prediction masks and formats them into a competition-ready submission.csv file using Run-Length Encoding (RLE).

## Final Results
The trained U-Net model achieved a Dice Coefficient of 0.68 on the unseen test set, demonstrating its ability to generalize and accurately segment glandular structures.

### Example Prediction:
Here is a sample prediction from the test set, with the model's predicted mask overlaid in purple.

## Technology Stack
Frameworks: TensorFlow, Keras

Data Handling: Deeplake (for cloud data streaming), NumPy

Image Processing: OpenCV, Pillow

Visualization: Matplotlib

Machine Learning: Scikit-learn

## Project Structure
The repository is organized into several key scripts that form the project pipeline:

File

Description

data.py

Data Processor: Loads the GlaS dataset from Deeplake, preprocesses images and masks to 256x256, and saves them as .npy files.

train.py

Model Trainer: Loads the processed data, builds the U-Net model, trains it with data augmentation, and saves the final predictions.

submissions.py

Submission Formatter: Loads the model's predictions and converts them into the Run-Length Encoded (RLE) format for the submission.csv file.

evaluate.py

Model Evaluator: Calculates the final Dice Coefficient of the model on the unseen test set by comparing predictions to the ground-truth masks.

visualize.py

Result Visualizer: Displays a random sample of test images with the model's predicted segmentation masks overlaid for qualitative assessment.

requirements.txt

Lists all the Python dependencies required to run the project.

## Setup and Usage
To run this project, follow these steps:

### 1. Prerequisites
Python 3.8+

Git

### 2. Clone the Repository
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

### 3. Set Up a Virtual Environment (Recommended)
Create the virtual environment
python -m venv venv

Activate it
On Windows:
.\venv\Scripts\Activate
On macOS/Linux:
source venv/bin/activate

### 4. Install Dependencies
Install all the required libraries from the requirements.txt file.

pip install -r requirements.txt

### 5. Run the Pipeline
Execute the scripts in the following order. Each script builds upon the output of the previous one.

Step 1: Process the Data
This script will download the dataset in the background and create the necessary .npy files.

python data.py

Step 2: Train the Model
This will train the U-Net and save the best model weights (weights.keras) and the final predictions (imgs_mask_test_predicted.npy).

python train.py

Step 3: Evaluate Performance (Optional)
To get the final Dice Score on the test set, run:

python evaluate.py

Step 4: Generate Submission File (Optional)
To create the submission.csv file with RLE, run:

python submissions.py

Step 5: Visualize Results (Optional)
To see your model's predictions overlaid on the test images, run:

python visualize.py
