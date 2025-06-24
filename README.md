# DEEP-LEARNING-PROJECT
COMPANY: CODTECH IT SOLUTIONS

NAME: IPPILI NITESH

INTERN ID: CT06DL1202

DOMAIN: DATA SCIENCE

DURATION: 6 WEEKS

MENTOR: NEELA SANTHOSH

# Description
In this task, a Convolutional Neural Network (CNN) model was implemented using TensorFlow to classify images from the CIFAR-10 dataset. The model was trained, evaluated, and visualized, showcasing a complete deep learning workflow. This project fulfills the requirement of building a deep learning model with visual outputs.

Steps Performed
Step 1: Load the Dataset
The CIFAR-10 dataset was loaded using tf.keras.datasets. It contains 60,000 32x32 color images in 10 categories, with 50,000 for training and 10,000 for testing.

Step 2: Normalize and One-Hot Encode
Pixel values were normalized between 0 and 1 for faster training. The target labels were one-hot encoded to match the output layer of the model.

Step 3: Build the CNN Model
A deep CNN architecture was created with multiple Conv2D, MaxPooling, and Dense layers, including a Dropout layer to reduce overfitting.

Step 4: Compile and Train
The model was compiled using the Adam optimizer and categorical crossentropy loss. It was trained over 15 epochs with a batch size of 64 and validated on the test dataset.

Step 5: Visualize Accuracy and Loss
Training and validation accuracy/loss curves were plotted over each epoch to evaluate learning performance.

Step 6: Evaluate and Predict
Model performance was evaluated on the test set. Sample predictions were visualized alongside actual labels to show model output correctness.

# OUTPUT
