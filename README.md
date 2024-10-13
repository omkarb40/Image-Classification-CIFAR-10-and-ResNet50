# CIFAR-10 Image Classification using Transfer Learning with ResNet50

This project showcases an image classification model built using transfer learning on the CIFAR-10 dataset. We leverage a pre-trained ResNet50 model, fine-tune it on CIFAR-10, and enhance performance using data augmentation techniques. The project aims to demonstrate advanced deep learning techniques suitable for real-world applications.

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 categories, including objects like airplanes, automobiles, birds, cats, and more. The project uses a Convolutional Neural Network (CNN) based on ResNet50, a powerful deep learning model pre-trained on the ImageNet dataset. Transfer learning allows us to utilize the pre-trained weights and adapt them to the CIFAR-10 dataset with minimal computation, achieving strong classification performance.

## Project Steps

### 1. Data Loading and Preprocessing
   - Loaded the CIFAR-10 dataset directly using TensorFlow/Keras.
   - Normalized pixel values to range between 0 and 1 to aid efficient training.

### 2. Data Augmentation
   - Applied data augmentation techniques such as horizontal flipping, random rotation, and zooming.
   - Augmentation increases the diversity of the dataset, improving the model’s generalization ability.

### 3. Transfer Learning with ResNet50
   - Initialized the ResNet50 model pre-trained on ImageNet, excluding the top (fully connected) layers.
   - Added custom dense layers to adapt the model to CIFAR-10's 10 classes.
   - Initially froze the base model layers to retain learned features from ImageNet.

### 4. Model Compilation and Training
   - Compiled the model using the Adam optimizer and sparse categorical cross-entropy loss.
   - Trained the model for 10 epochs, validating performance on the test dataset.
   
### 5. Fine-Tuning
   - Unfroze the pre-trained ResNet50 layers and continued training with a reduced learning rate.
   - Fine-tuning adjusts high-level feature representations in the pre-trained model, optimizing it further for CIFAR-10.

### 6. Evaluation and Visualization
   - Evaluated the model's accuracy on the test set to gauge real-world performance.
   - Visualized training and validation accuracy over epochs, providing insights into the model's learning process and improvements from fine-tuning.

## Results
   - The model achieves competitive accuracy on the CIFAR-10 dataset, demonstrating the effectiveness of transfer learning and data augmentation.
   - Fine-tuning further improves accuracy, showcasing the adaptability of pre-trained models.

## Project Structure

```plaintext
├── cifar10_image_classification.ipynb   # Jupyter Notebook with the project code
├── README.md                            # Project README with explanation and details
└── images/                              # Folder for storing sample images and plots
