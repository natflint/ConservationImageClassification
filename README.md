# Classifying Animals in Convservation Park Using Deep Learning Methods
The introduction of trap cameras to wildlife conservationist activities has allowed for conservationists to track animals in their natural habitats, while leaving them undisturbed. In this study we will attempt to build an image classification pipeline using PyTorch deep learning neural networks to classify images taken on trap camera images. However, through the development and investigation of our deep learning model with Grad-CAMs, we discovered that the images are too much background noise from the images’ rainforest environment to successfully extract the features needed to classify these animals.

# Image Analysis and Processing:
There are 8 image classes within the dataset, including blank. They are as follows: 0 – antelope_duiker, 1 – bird, 2 – blank, 3 – civet_genet, 4 –  hog, 5 – leopard, 6 – monkey_prosimian, and 7 – rodent. 

In the first investigation of the images, the lack of actual colour images was discovered. There were 14,291 of the total 17,068 images that were structured as coloured images, but through further investigation only 4,296 of those 14,291 images were actually colored. Because of this, the model was converted to only accept grayscale images and in the data loader included a transformation to convert the images grayscale. In addition to the the grayscale transformation, the images were resized to 224 x 224, and data augmentation was performed through the use of random rotation, random perspective, random vertical flip, and random affine, as well as all images being normalised.

There were 148 unique sites where the images were taken within the park. The amount of images taken at each site ranged from a single image to up to 1,132 images. In looking at the sites, they are all similar, with lots of green vegetation, branches, and trees. However, some are more unique, with cameras clearly being pointed at the ground and others not being able to see any ground.

As with all camera trap images, there are a number of images where animals are only partially in the frame. Since the images were taken within a densely vegetated rainforest, when an image had an animal in it, the animal is oftentimes obstructed by the vegetation, so the animal is not fully in view. In other images, there is so much interference from the vegetation that the animals blend in with the background. On top of all this, the images themselves are of the best quality. Exaples of poor image quality and obstructed animals can be seen below.

 ![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/84b8592a-b334-415d-95ad-7216005a45b5) 
 ![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/f6589f4a-8830-4bea-a16d-e86e3e4f11a2)
 ![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/3bfd6700-6bea-42ab-b7d1-8850735de3ec)
 ![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/d2702f29-b64c-4995-b708-093bf6c579dc)

# Implemenation and Network Architecture Comparison

Four different common deep learning architectures for image classification were chosen for comparison, based on similar studies using camera trap images. These were AlexNet, VGG, GoogLeNet, and ResNet.

For each model architecture expiremented with, both the pretrained weights from ImageNet and no pretrained weights were expiremented with, though in the end our final version of each model used the pretrained weights. For each model, the architecture for the first convolutional layer needed to be adjusted for to accept grayscale images, as well as the last layer, for the eight possible output classes. 

# Results

The best model produced during training was ResNet. However, it is important to note that in all of these models all different sorts combinations of parameters ran with the models would almost bottom out at around the same score. This likely meant the models were hyperfitting the training data.

When the final trained model was applied to the holdout out data, the cross entropy score was 2.0701. This was not a good score. In further investigation of the holdout data, we discovered the model was consistently classifying each image as a civet genet(3). This was obviously a poor result, and as we had continously had poor scores through the training process, we decieded to investigate the model further. 

# Investigation Using Grad-CAM

Because of the poor score from our final model and throughout the training process, we deciede to investigate using Grad-CAMs. The decision to use Grad-CAMs specifically to see how the dense vegetation from the rainforest model affected feature extraction. To do this, we used the implemented Captum’s GuidedGrad-CAM feature for each class, specifically for the last convolutional layer of our ResNet model, to see what was extracted from images right before they were classified. Below, we have the original images as well as the output of the gradcams layered over the orginal images. 

![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/8319e82a-61a5-45f6-acdf-a9a9418915ac)

![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/a14e7566-9343-460b-8379-26352f9ca224)

By looking at these images, it can be confirmed that the model in fact did not learn the features of the animals, but instead the features of the background and environment. By looking at the images, it can be seen that for some images the area where the most information was extracted from did not even include the animal, such as for civet genet. For others, we can see that a little bit of information was taken from everywhere in the image, such as antelope duiker. 

In further investigation from some of the previously reviewed literature, as well as the images taken at different sites around the park, we discovered that in addition to the dense vegetation, the ratio of sites where the images were taken to the few number of images per each site, likely also played a role in the poorly scoring model. This was supported by Schneider, et al., that spoke on how “deep learning models will only reflect locations that were seen during training and will underperform at new locations” (Schneider, et al., 2020). Given that the 17,000 images came from 148 unique sites, there would only be a little over 100 images per site. For creating a deep learning neural network for image classification, this is not a sufficient number of images for a model to train on. So the issue of overwhelming vegetation from the environment becomes aided by the problem that there are too many unique sites, with not enough images coming from an even distribution of the sites.

# References
Schneider, S., Greenberg, S., Taylor, G. W., & Kremer, S. C. (2020). Three critical factors affecting automated image species recognition performance for camera traps. Ecology and Evolution, 10(7), 3503–3517. https://doi.org/10.1002/ece3.6147
https://captum.ai/api/guided_grad_cam.html
