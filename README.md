# Classifying Animals in Convservation Park Using Deep Learning Methods
The introduction of trap cameras to wildlife conservationist activities has allowed for conservationists to track animals in their natural habitats, while leaving them undisturbed. In this study we will attempt to build an image classification pipeline using PyTorch deep learning neural networks to classify images taken on trap camera images. However, through the development and investigation of our deep learning model with Grad-CAMs, we discovered that the images are too much background noise from the images’ rainforest environment to successfully extract the features needed to classify these animals.

# Image Analysis and Processing:
There are 8 image classes within the dataset, including blank. They are as follows: 0 – antelope_duiker, 1 – bird, 2 – blank, 3 – civet_genet, 4 –  hog, 5 – leopard, 6 – monkey_prosimian, and 7 – rodent. 

In the first investigation of the images, we discovedred the lack of actual colour images. In our oringal invesitagiont we discovered that 14,291 of the total 17,068 images were structured as coloured images, but through further investigation only 4,296 images were actually colored. Because only about a fourth of all images were coloured, the model was converted to only accept grayscale images and in the data loader included a transformation to convert the images grayscale. In addition to the the grayscale transformation, the images were resized to 224 x 224, and data augmentation was performed through the use of random rotation, random perspective, random vertical flip, and random affine, as well as all images being normalised.

There are 148 unique sites where the images were taken within the park. The amount of images taken at each site ranges from a single image up to 1,132 images. In looking at the sites, they are all similar, with lots of green vegetation, branches, and trees. However, some are more unique, with cameras clearly being pointed at the ground and others not being able to see any ground.

As with all camera trap images, there are a number of images where animals are only partially in the frame. Since the images were taken within a densely vegetated rainforest, when an image has an animal in it, the animal is oftentimes obstructed by the vegetation, so the animal is not fully in view. In other images, there is so much interference from the vegetation that the animals blend in with the background. On top of all this, the images themselves are of the best quality. Exaples of poor image quality and obstructed animals can be seen below.

<img src="(https://github.com/natflint/ConservationImageClassification/assets/115076736/84b8592a-b334-415d-95ad-7216005a45b5)" width="425"/> <img src="(https://github.com/natflint/ConservationImageClassification/assets/115076736/f6589f4a-8830-4bea-a16d-e86e3e4f11a2)" width="425"/> 


 ![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/84b8592a-b334-415d-95ad-7216005a45b5) *a night image with flash bounce back*
 ![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/f6589f4a-8830-4bea-a16d-e86e3e4f11a2)
 ![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/3bfd6700-6bea-42ab-b7d1-8850735de3ec)
 ![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/d2702f29-b64c-4995-b708-093bf6c579dc)

# Implemenation and Network Architecture Comparison



# Results

# Investigation Using Grad-CAM
![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/8319e82a-61a5-45f6-acdf-a9a9418915ac)

![image](https://github.com/natflint/ConservationImageClassification/assets/115076736/a14e7566-9343-460b-8379-26352f9ca224)


