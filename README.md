# Classifying Animals in Convservation Park Using Deep Learning Methods
The introduction of trap cameras to wildlife conservationist activities has allowed for conservationists to track animals in their natural habitats, while leaving them undisturbed. In this study we will attempt to build an image classification pipeline using PyTorch deep learning neural networks to classify images taken on trap camera images. However, through the development and investigation of our deep learning model with Grad-CAMs, we discovered that the images are too much background noise from the images’ rainforest environment to successfully extract the features needed to classify these animals.

# Image Analysis and Processing:
The first thing we dealt with for manipulation was converting the one-hot encoded labels for the 8 possible classes to be categorically encoded, meaning all 0-7 labels were in the same column, “class”.  The categories and their corresponding numbers are as follows:  0 – antelope_duiker, 1 – bird, 2 – blank, 3 – civet_genet, 4 –  hog, 5 – leopard, 6 – monkey_prosimian, and 7 – rodent. This was done for the image classification architectures in the deep learning models in PyTorch.

The lack of true colour images did bring up another issue. Because only about a fourth of all images were coloured, we did not think it would be a wise decision to fully train the model as if all of the images were RGB. So when creating our custom data loader for Pytorch, we decided to convert the images to grayscale. 
 ![A night image with flash bounce back](https://github.com/natflint/ConservationImageClassification/assets/115076736/84b8592a-b334-415d-95ad-7216005a45b5)  
 ![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/f6589f4a-8830-4bea-a16d-e86e3e4f11a2)
 ![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/3bfd6700-6bea-42ab-b7d1-8850735de3ec)
 ![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/d2702f29-b64c-4995-b708-093bf6c579dc)

# Implemenation and Network Architecture Comparison

# Results

# Investigation Using Grad-CAM


