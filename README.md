# Image Recognition - Object Classification AI  #

The purpose of this project is to create an object recogniser with the help of PyTorch models. 

## Motivation ##

Images can contain a large variety of objects, which can range anywhere from humans to cars. These objects can be recognised 
and classified through the use of artificial intelligence. The purpose of this object recogniser is to provide people of the
public with a tool that is easily able to recognise different objects in an image, regardless of their location in the image.

## What Does It Do? ##
This project is able to recognise objects in an image through the use of four pytorch models. The models available in this project are; AlexNet, SENet, ResNet and VGG. Using the main.py file provided, the user can choose which model they would like to use by inputting the corresponding number.

## Testing Environment ##
This project uses the CIFAR-10 dataset for the training and testing dataset. The models presented in this project were tested with this dataset on two seperate environments. The results highlighted here have been 
obtained in these environments. The details for the testing environments 
are listed below. <br />
### Environment 1: <br />
CPU: Intel(R) Core(™) i7 1.8GHz <br />
GPU: Intel(R) UHD Graphics 620 <br />
Memory: 8.0 GB <br />
IDE: Pycharm <br />
### Environment 2: <br />
CPU: Intel(R) Core(™) i5 1.8GHz <br />
GPU: Intel HD Graphics 6000 1536 MB <br />
Memory: 8.0 GB <br />
IDE: Pycharm <br />

## Software Required ##
The list below shows the tools that were used in this project and where they can be downloaded from. <br />
Python 3.7 - https://www.python.org/downloads/release/python-377/ <br />
Anaconda 3 - https://www.anaconda.com/products/individual <br />
PyTorch - https://pytorch.org/get-started/locally/
###### If wanting to used CUDA, make sure GPU in your computer is by NVIDIA ######
Follow the instructions provided on the links given and create a Conda environment.

## Run Code ##
To be able to run this code in your command prompt, you can clone this repository or download the zip file. To clone and 
download this repository, the option can be found on the right side of the repository page. Cloning can also be done by 
using the command, <br />
``` 
$ git clone https://github.com/cng822/Image-Recognition-AI.git 
$ cd Image-Recognition-AI
$ main.py
```
If wanting to run the code through an IDE, ensure the IDE is Python compatible. Only the main.py file needs to be run. 
## Results From Models Provided ##
| Model Used | Accuracy Obtained |
| ---------- | ----------------- |
| AlexNet    | 70%               |
| ResNet     | 85%               |
| SENet      | 85%               |
| VGG        | 85%               |



 



