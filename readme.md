# AI Waste Sorting using transfer learning with Pytorch
U4 Project - Deep Learning for Business - AI&BA - TBS
Antoine Settelen, Edgar Jullien, Simon Weiss   

~3 weeks development  


Check the demo [here](https://ml-app-pytorch.herokuapp.com/).



## Introduction

In this project we will develop a classifier to seperate waste into six classes : glass, paper, cardboard, plastic, metal, and trash based on their images. As a result of rapid urbanization and population growth, the amount of waste produced each year in the world is expected to rise to 3.4 billion tons over the next three decades, up from 2.01 billion tons in 2016, according to the World Bank.   
What a Waste 2.0 report underlines the crucial importance of household waste management for sustainable, healthy and inclusive urban development, and highlights the fact that this sector is often neglected, particularly in low-income countries.
This project aims to demonstrate the potential of AI in waste management and in autoamtic sorting systems helping ciruclar economies to better design product reuse and recycling.  

![Recycling Waste](https://i.pinimg.com/originals/d1/89/cb/d189cbde475f5994917c1ed32fb8b1e0.jpg)

Automatically classifying the types of waste would effectively : 
- be an aid to the domestic sorting of waste
- allow for a verification system at the time of waste collection
- be integrated into sorting machines in recycling plants 
- Reduce toxic waste ending in landfills

### Dataset

The data comes from the dataset [trashnet](https://github.com/garythung/trashnet) for a final project of [Stanford's CS 229: Machine Learning class](http://cs229.stanford.edu)
the dataset consists of 2527 images:
- 501 glass
- 594 paper
- 403 cardboard
- 482 plastic
- 410 metal
- 137 trash

The pictures were taken by placing the object on a white posterboard and using sunlight and/or room lighting. The pictures have been resized down to 512 x 384, which can be changed in `data/constants.py` (resizing them involves going through step 1 in usage). The devices used were Apple iPhone 7 Plus, Apple iPhone 5S, and Apple iPhone SE.
You can find the dataset used in the /dataset folder inside the notebook folder. 

### Pytorch Learning
You will find our notebook used to train our model in /notebook folder and its html generated file. You can open it locally our use colab to use GPU instance provided by Google. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-rC0ingheeVI6kjBm8YBEk9q2NPm3nB6#scrollTo=P1b-eEtXq6cT)  
First, we configured our own neural network with pytorch and then tested our results with the accuracy score. 
Then we used ResNet-50 CNN within the transfer learning method. Our maximum score was 97% accuracy. 
ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images.
Once our model was trained, we exported its weights as a pth file "cnn2.pth" to use it in our web version deployed on heroku. 

## Demo

Check the demo [here](https://ml-app-pytorch.herokuapp.com/). You can use the dataset images for testing but also use your own images. Our model is not perfect and is only the beginning of neural network definition, so it will certainly have incorect predictions on a custom dataset.   
We will appreciate any recommendations or advices to improve our model! 

### Screenshots

![Home](https://user-images.githubusercontent.com/640792/63314664-09142700-c326-11e9-90fc-ae358c59b045.jpg)

![Classifiction](https://user-images.githubusercontent.com/640792/63314665-09142700-c326-11e9-9075-607a5d900bd1.jpg)


## Installation and Deployment


### System Requirements


Works with Python 3.5 and above


### Python Requirements

Install them from `requirements.txt`:

    pip install -r requirements.txt


### Local Deployment

Run the server:

    python app.py


Visit `localhost:5000`

### Heroku Deployment

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/avinassh/breast-cancer-prediction)



## Acknowledgments
- Thanks to the Stanford CS 229 autumn 2016-2017 teaching staff for a great class!
- [@e-lab](http://github.com/e-lab) for their [weight-init Torch module](http://github.com/e-lab/torch-toolbox/blob/master/Weight-init/weight-init.lua)

Test
Inspire by github repo : 
https://github.com/imadelh/ML-web-app
https://github.com/imadtoubal/Pytorch-Flask-Starter

link : https://ai-sorting.herokuapp.com/
