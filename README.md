# 🎉 Supervisely : Supercharge your training data Pipeline with Deep Learning

# Supervisely
Supervisely helps companies, researchers, engineers, students, and many others make and prepare training data for various number of computer vision tasks from pedestrian detection to tumors segmentation.
![alt text](https://miro.medium.com/max/700/1*a5uGXTbN6_AdUXAiWFIRWg.png)

# Objective
Create a project designed to solve the real use case, using either transfer learning example existing Mask-RCNN, VGG16, etc. or creating new model of Mask-RCNN, GANs, RNN, etc. to solve any real case problems or new problems. 

# Necessary requirements:
1.  Make your own custom dataset using supervisely 
2.  Either create a new model or using existing model as transfer learning
3.  Launch the training on aws cloud

# What exactly is Supervisely?
Deep Learning is here to stay. The more training data — the smarter AI. Our mission is to provide the companies with the tools to perform image annotation as efficient as possible.
1. The first use case is pretty simple. Upload your data and start manual annotation with our AI powered tools that specifically addresses the annotation process for semantic segmentation task. When annotation is finished, you can download images and annotations in desirable format.
2. The second use case illustrates the usage of pretrained neural network from Model Zoo. After user has uploaded images he can apply neural network to dataset for pre-annotation. There are a large number of examples, where ready to use models can be used for speeding up annotation process: i.e. detecting workers in construction areas with bounding boxes or segmenting humans on selfie images.
3. The third use case demonstrates human-in-the-loop AI approach. User can train neural network for his custom task and then use this NN to pre-annotate images. Then user should only correct NN predictions. And this process is iterative. NN becomes smarter with time, annotated data is increased, and the process of annotation is accelerating, and user can repeat this procedure over and over again until necessary accuracy is obtained. As a result user get both big dataset with high-quality annotations and accurate NN for his own specific task.

![alt text](https://miro.medium.com/max/700/1*d2K9ptzVIFq5GBCSwGCf2g.png)

First step to start with this demo you should have your own supervise.ly account.
After registering you will find plenty of resources on the website like Annotation tools, Labelling tool, Classification, and also Jupyter Notebooks like scripting material resources. But for our project, We have to annotate the dataset first. Hence, We required to import the dataset first by creating a workspace in New Project.
![alt text](https://github.com/BS-8100/Supervisely/blob/master/2.JPG)
![alt text](https://github.com/BS-8100/Supervisely/blob/master/27.JPG)
![alt text](https://github.com/BS-8100/Supervisely/blob/master/28.JPG)

# Data Preparation
# Data Annotation
To train Mask R-CNN we will use our tiny dataset containing only 6 images. In each image there are several annotated fruits belonging to different classes.To train such a deep NN we have to prepare training dataset: perform various data augmentations. Supervisely has Data Transformation Language (DTL) specially designed for that purpose.
![alt text](https://github.com/BS-8100/Supervisely/blob/master/29.JPG)

 Full DTL config:
 
 ```
 [
  {
    "dst": "$sample",
    "src": [
      "lemon/*"
    ],
    "action": "data",
    "settings": {
      "classes_mapping": "default"
    }
  },
  {
    "dst": "$fv",
    "src": [
      "$sample"
    ],
    "action": "flip",
    "settings": {
      "axis": "vertical"
    }
  },
  {
    "dst": "$fh",
    "src": [
      "$fv",
      "$sample"
    ],
    "action": "flip",
    "settings": {
      "axis": "horizontal"
    }
  },
  {
    "dst": "$data",
    "src": [
      "$fv",
      "$sample",
      "$fh"
    ],
    "action": "dummy",
    "settings": {}
  },
  {
    "dst": "$data2",
    "src": [
      "$data"
    ],
    "action": "multiply",
    "settings": {
      "multiply": 10
    }
  },
  {
    "dst": "$data3",
    "src": [
      "$data2"
    ],
    "action": "crop",
    "settings": {
      "random_part": {
        "width": {
          "max_percent": 90,
          "min_percent": 70
        },
        "height": {
          "max_percent": 90,
          "min_percent": 70
        },
        "keep_aspect_ratio": false
      }
    }
  },
  {
    "dst": [
      "$totrain",
      "$toval"
    ],
    "src": [
      "$data3",
      "$data"
    ],
    "action": "if",
    "settings": {
      "condition": {
        "probability": 0.95
      }
    }
  },
  {
    "dst": "$train",
    "src": [
      "$totrain"
    ],
    "action": "tag",
    "settings": {
      "tag": "train",
      "action": "add"
    }
  },
  {
    "dst": "$val",
    "src": [
      "$toval"
    ],
    "action": "tag",
    "settings": {
      "tag": "val",
      "action": "add"
    }
  },
  {
    "dst": "lemon_train",
    "src": [
      "$train",
      "$val"
    ],
    "action": "supervisely",
    "settings": {}
  }
]
```
Computational graph:
![alt text](https://github.com/BS-8100/Supervisely/blob/master/mask_02.png)

## Add NN architecture and pretrained weights¶
To add a new architecture with pretrained weights to your account you should go to Explore -> Models. Find Mask-RCNN, click Add and then Clone.
![alt text](https://github.com/BS-8100/Supervisely/blob/master/21.JPG)

From Menu: Neural Network → Click on Add → Plugin → No agents Available
So, You are thinking currently why this error?
Simple Reason, Supervise.ly doesn’t provide GPU to train the models hence you have to use Local System or Cloud Services for the same.
![alt text](https://github.com/BS-8100/Supervisely/blob/master/22.JPG)

The solution for this is training the model using the GPU instances over Amazon AWS.
First, you have to create an Amazon AWS account and after registration has been done. Follow the below steps:
Click on Services → EC2 → Create an Instance → Choose AMI → Type

![alt text](https://github.com/BS-8100/Supervisely/blob/master/23.JPG)
![alt text](https://github.com/BS-8100/Supervisely/blob/master/24.JPG)
![alt text](https://github.com/BS-8100/Supervisely/blob/master/25.JPG)
If you are a new user, You will face one problem in instance creation after completing all the above steps successfully.
ERROR
It’s not the error logically because new users are assigned some limits and if you are trying to use more than 1 vCPU then you have to create a support case for requesting a Limit increase to Amazon AWS support center.
![alt text](https://github.com/BS-8100/Supervisely/blob/master/26.JPG)
