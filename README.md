# Udacity Robotics Project 4 - Follow Me

In this project, I'll train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques I apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./misc/simulator.png
[image_1]: ./misc/points.png
[image_2]: ./misc/people.png
[image_3]: ./misc/model1_IOU.png
[image_4]: ./misc/train_curve_model1.png
[image_5]: ./misc/model2_64_IOU.png
[image_6]: ./misc/train_curve_model2_64.png
[image_7]: ./misc/model2_96_IOU.png
[image_8]: ./misc/train_curve_model2_96.png
[image_9]: ./misc/model3_32_IOU.png
[image_10]: ./misc/train_curve_model3_32.png
[image_11]: ./misc/model3_96_IOU.png
[image_12]: ./misc/train_curve_model3_96.png
[image_13]: ./misc/melt.png
[image_14]: ./misc/model1_hand.png
[image_15]: ./misc/model2_hand.png
[image_16]: ./misc/model3_hand.png


## Simulator
The Quad-Copter Simulator as below, and you can download it [here](https://github.com/udacity/RoboND-DeepLearning-Project/releases/tag/v1.2.2):
![alt text][image_0]   
![alt text][image_1]  
![alt text][image_2]  
Datas which are used for FCN to train the model can get from the "DL Training" mode in the Simulator.
* Patrol points: The quadcopter can navigate along the path with patrol points.
* Hero Path Points: Hero is who called the quadcopter to follow, hero will step along "hero path points".
* Spawn Points: Other people will go around with "Spawn Points"

## FCN Network
* FCN: I use the FCN architecture to segment objects. FCN is an encoder-decoder architecture and use Skip Connection to generate margin of object. So with FCN I can identify which one is the hero the Copter will follow.
* Encoder-Dcoder: The encoder is a usual convolution network to find object features and tell us which object in the current data. The encoder can  The decoder is Transposed Convolution Layer, it use the feature encoder output to generate general-purpose view of object for example margin. The layer connected encoder and decoder is 1x1 convolution layer, this is different from normal Convlution Network which use fully connected layers to be the output layer. All of encoder and decoder will use Separable Convolution Layer to be the base Convolution layer.
* Why No Fully Connected Layers: Normal CNN Network can identify the feature of inputs. Feature in images means what's the shape, color or other profile of objects. CNN is good technology to do classification, for example identify the specified object is in the current image or not. But the question "where is the object in the image?" will be ignore, because the output layer of CNN is fully connected layers, this kind of layer usually make features into a array. It will lost the information about space and only remain existence. So use 1x1 convolution layers to instead of fully connected layers can retain important information for segmentation.
* Skip Connection: When images step into FCN network, they will get change in the network piplines, I can say "melt". The pipline is longer,  the feature seems more melting. So if I grep the feature from eraly layers and concatenate it to the later layses, it will be better. That's like people to remember a new word. When my son has seen a new word yesterday and now meet it again, He can easily to remember it. This technology called "skip connection". Below picture shows the effect, the 1st picture from left is no skip connection, next one is the effect to add one skip connection from previous layer. You can see the margin of objects will be more clearly with the number of skip connection increase:
![alt text][image_13] 
* Separable Convolutions: Focus on the encoder-decoder architecture. I use [Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf) instead of traditional conv2d+maxpool layers, it's useful to reduce parameters and running time. With help of some codes in \utils I can easily define the encoder and decoder layers with Keras as below:
* Batch Normalization: [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf) is a important technology to get high validation accuracy. By using the mean and variance of the values in the current mini-batch I can get good inputs. Also Normalize the input batch to every layers is the best way to improve accuracy. 
* Bilinear Upsampling: Here Bilinear Upsampling is a resampling technique which served for transpose convolutions(de-convolution). Do Bilinear Upsampling with small layer will make them to be the original size. But there are also some informations will be lost unless use skip connections(layers.concatenate) to retain information from previous laysers.

## Models
* I tried 3 kinds of models with different layer architecture.
 - Model 1 has two encoders and decoders.
 - Model 2 has three encoders and decoders.
 - Model 3 has five encoders and decoders.
* About parameters
 - All Separable Convolutions are set kernel_size as 3, set padding as 'same' to keep the shape, set activation as 'relu'.
 - I use 96 for started fiters of model_1.
 - I use 64 and 96 for started fiters of model_2, There are two configuration.
 - I use 32 and 96 for started fiters of model_3, There are two configuration.
* Some results as below:

|  |model_1 | model_2_64 | model_2_96 | model_3_32 | model_3_96
|- | - | - | - | - | -
|Train Loss | 0.0160 | 0.0114 | 0.0122 | 0.0120 | 0.0097 
|Var Loss | 0.0439 | 0.0319 | 0.0240 | 0.0256 | 0.0273 
|IOU | 0.4247 | 0.5117 | 0.5888 | 0.5456 | 0.5630 
|Score | 0.3149 | 0.3707 | 0.4390 | 0.3942 | 0.4265 

## My Model Architecture
I use keras.utils.vis_utils.plot_model to plot the model as png image.
* model 1
![alt text][image_14]

* model 2 (This is the model in my final code, two config, I use 96 to be started filter number.)
![alt text][image_15]

* model 3
![alt text][image_16]


## Hyperparameters
 - Epoch : 50 is my choise to balance the loss-decay and trainning-time.
 - Learning Rate: All my trainning process with learning_rate as 0.03, the loss can arrive a good value and no overfitting happen. So I feel it's OK.
 - Batch Size: I set batch_size to be 32, becuase the trainning process I did with a Nvidia GTX1080, 8G GPU Ram.
 - Number of workers: The default 2 worker is just fine, the trainning process will not exceed 20 minutes.
 - Steps per epoch: 200 step means 200*32 images in 1 epoch, this is OK for default datas which has 4131 images, most images will be used to train the model.
 - Validation steps: 50 step means 50*32 images in 1 epoch, this is almost fine for 1184 images in the default validation datas.


## Trainning Result
I show my result below, these result may change because of random data in every trainning epoch.
### Result of model_1
![alt text][image_3] 
![alt text][image_4] 

### Result of model_2_64
![alt text][image_5] 
![alt text][image_6] 

### Result of model_2_96(This is which I submit for project)
![alt text][image_7] 
![alt text][image_8] 

### Result of model_3_32
![alt text][image_9] 
![alt text][image_10] 

### Result of model_3_96
![alt text][image_11] 
![alt text][image_12] 


## Future Enhancement
* This encoder-decoder architecture FCN has can identify and locate objects in complex scenes. But this model can't percept and trace other object(dog, cat, car, etc.), because there is no feature about them. So adding more datas as like other images or more pix(height and width) will improve the model to be generalization. On the other hand , more datas need more complex model. Therefore I suggest to use wights of ImageNet trained by ResNet as beginning wights.
* The performance of the network seemed good enough. Just because the one the Copter followed wears red clothes, this is easy to identify her because none of other people wear red clothes. The groud, sky, environment also has little red color pix. It'll be hard to follow other people by this model. Maybe some other technology as like body gesture recognization can solve this problem.