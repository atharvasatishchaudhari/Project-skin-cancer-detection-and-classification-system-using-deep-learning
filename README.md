# Project: skin-cancer-detection-and-classification-system-using-deep-learning

# Aim
The main goal of this project is to develop a deep learning-based system for early detection and classification of skin cancer from dermatoscopic images. I focused on two key tasks: <br>
1)Skin Lesion Segmentation: Isolating the lesion area from the input images using a deep segmentation model. <br>
2)Skin Cancer Classification: Categorizing the segmented lesions into different skin cancer types using CNN models. <br>

# Process
Data Collection & Preprocessing
Dataset: I used the HAM10000 dataset, which consists of a large number of dermatoscopic images. <br>
Preprocessing: I applied data augmentation techniques (such as flipping, cropping, and brightness adjustments) and normalized the images to a uniform size. These steps help in increasing the dataset size and reducing overfitting.

# Skin Lesion Segmentation
I implemented a Bidirectional Convolutional U-Net (BCDU-Net) model to accurately segment the skin lesions. <br>
The segmentation quality was evaluated using metrics like Dice Coefficient (90.66%) and Intersection over Union (IOU: 83.09%). 

# Skin Cancer Classification
For the classification task, I experimented with two CNN architectures: VGG-19 and DenseNet121. <br>
I leveraged transfer learning and used callbacks to monitor and optimize the training process.
The VGG-19 model achieved outstanding performance, while the DenseNet121 (implemented alongside VGG-19) delivered slightly lower accuracy (around 88–89%).

# Results
Segmentation:<br>
Dice Coefficient: 90.66% <br>
IOU: 83.09% <br>

Classification (VGG-19): <br>
Accuracy: 97.29% <br>
Precision: 97.42% <br>
Recall: 97.42% <br>
F1-Score: 97.42% <br>

Classification (DenseNet121): <br>
Accuracy: ~88–89% (compared to the higher accuracy of VGG-19)

# Technology Used
Language: Python <br>
Frameworks: TensorFlow, Keras <br>
Development Environment: Jupyter Notebook <br>
Libraries: OpenCV, NumPy, Pandas, Matplotlib <br>
Dataset: HAM10000 (Dermatoscopic skin lesion images) <br>

# File Breakdown
Skin cancer Detection.ipynb:
This notebook contains the code for skin lesion segmentation using the BCDU-Net model. It covers data preprocessing, segmentation, and evaluation using metrics like Dice Coefficient and IOU. <br>

vgg19.ipynb:
This file includes the implementation of the VGG-19 based classification model. It details the model architecture, training process, and evaluation results. <br>

skin-cancer-model-comparison-VGG19-DenseNet121.ipynb:
In this notebook, I implemented experiments for both VGG-19 and DenseNet121 models. DenseNet121 is implemented here alongside VGG-19, and I compare their performance. The results show that VGG-19 outperforms DenseNet121 in terms of accuracy and other evaluation metrics. <br>

# Publication
Present paper at International Conference on Computational Intelligence, Network and Security (IEEE) (22-23 December 2023) 
