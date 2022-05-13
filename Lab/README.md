# Laboration Deep Learning



## Transfer Learning : Inception 

### 1. Introduction : 

AlexNet, an image network that won the 2012 ImageNet best-of-breed competition, is also popularly used for different activities, such as object-detection, segmentation, human pose calculation, video type interpretation, object tracking, and superresolution.

These network architectures motivated research efforts that led to improved convolutional neural networks. Among these networks, VGGNet and GoogLeNet delivered results in 2014 that surpassed those of all other networks in ImageNet classification.

VGGNet has the benefit of being very simple to configure and implement, but its computational constraints need to be carefully assessed, and Inception doesn't.

Inception is also substantially less expensive than VGGNet and its higher performing successors. This allows it to operate effectively in big-data scenarios, where a large amount of information needs to be processed at low cost or in scenarios where memory and computational power is limited, for instance, in mobile vision.


### 2. General Design Principles :
### 3. Factorizing Convolutions with Large Filter Size : 
#### 3.1. Factorization into smaller convolutions :
#### 3.2. Spatial Factorization into Asymmetric Convolutions :
### 4. Utility of Auxiliary Classifiers :
### 5. Efficient Grid Size Reduction :
### 6. Inception-v2 :
### 7. Model Regularization via Label Smoothing :
### 8. Training Methodology :
### 9. Performance on Lower Resolution Input :
### 10. Experimental Results and Comparisons :
### 11. Conclusions :
