# Udacity Robotics Project 4 - Follow Me

In this project, I will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.
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
* FCN: I use the FCN architecture to segment objects. FCN is an encoder-decoder architecture, encoder is a usual convolution network but the output with a 1x1 convolution layer instead of fully connected layers. Compared to the fully connected layer which would not preserve the spatial information, 1x1 convolution layer then follow de-convolution layers will keep more spatial information like the positon of object. So if we do classification, we can only use fully connected layers, but when we want to do object se  
* Separable Convolutions: Focus on the encoder-decoder architecture. I use [Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf) instead of traditional conv2d+maxpool layers, it's useful to reduce parameters and running time. With help of some codes in \utils I can easily define the encoder and decoder layers with Keras as below:

 - *Separable Convolution Layer with Batch Normalization*
<code>
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
&ensp;&ensp;&ensp;&ensp;output_layer_1 = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,padding='same', activation='relu')(input_layer)   
&ensp;&ensp;&ensp;&ensp;output_layer = layers.BatchNormalization()(output_layer_1) 
&ensp;&ensp;&ensp;&ensp;return output_layer
</code>
<br>
 - *Normal Convolution Layer with Batch Normalization*
<code>
def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
&ensp;&ensp;&ensp;&ensp;output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(input_layer)
&ensp;&ensp;&ensp;&ensp;output_layer = layers.BatchNormalization()(output_layer) 
&ensp;&ensp;&ensp;&ensp;return output_layer
</code>
<br>
 - *Bilinear Upsampling*
<code>
def bilinear_upsample(input_layer):
&ensp;&ensp;&ensp;&ensp;output_layer = BilinearUpSampling2D((2,2))(input_layer)
&ensp;&ensp;&ensp;&ensp;return output_layer
</code> 
<br>
 - *Encoder Layer*
<code>
def encoder_block(input_layer, filters, strides):
&ensp;&ensp;&ensp;&ensp;output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
&ensp;&ensp;&ensp;&ensp;return output_layer
</code> 
<br>
 - *Decoder Layer*
<code>
def decoder_block(small_ip_layer, large_ip_layer, filters):
&ensp;&ensp;&ensp;&ensp;output_upsample = bilinear_upsample(small_ip_layer)
&ensp;&ensp;&ensp;&ensp;output_concatenate = layers.concatenate([output_upsample, large_ip_layer])
&ensp;&ensp;&ensp;&ensp;output_layer =  separable_conv2d_batchnorm(output_concatenate, filters, strides=1)
&ensp;&ensp;&ensp;&ensp;output_layer =  separable_conv2d_batchnorm(output_layer, filters, strides=1)
&ensp;&ensp;&ensp;&ensp;return output_layer
</code>
<br>
* Batch Normalization: [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf) is a important technology to get high validation accuracy. By using the mean and variance of the values in the current mini-batch I can get good inputs. Also do this to every layers(mean normalize the inputs to every layers) is the best way for trainning process. 
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

* About Hyperparameters
 - All my trainning process with learning_rate as 0.03, the loss can arrive a good value and no overfitting happen. So I feel it's OK.
 - I set batch_size to be 32, becuase the trainning process I did with a Nvidia GTX1080, 8G GPU Ram.
 - 50 epochs and 200 step will work effectively.

* Model e.g.
 - model_1: 
<code>
filter_begin = 96
encoder_1 = encoder_block(inputs, filters=filter_begin, strides=2)  
encoder_out = encoder_block(encoder_1, filters=filter_begin*2, strides=2)  
cov_1 = conv2d_batchnorm(encoder_out, filters=filter_begin*2, kernel_size=1, strides=1)  
decoder1 = decoder_block(cov_1, encoder_1, filters=filter_begin*2)  
decoder_out = decoder_block(decoder1, input, filters=filter_begin)
</code>
<br>

 - model_2_64: 
<code>
filter_begin = 64  
encoder_1 = encoder_block(inputs, filters=filter_begin, strides=2)  
encoder_2 = encoder_block(encoder_1, filters=filter_begin*2, strides=2)  
encoder_out = encoder_block(encoder_2, filters=filter_begin*4, strides=2)  
cov_1 = conv2d_batchnorm(encoder_out, filters=filter_begin*4, kernel_size=1, strides=1)  
decoder1 = decoder_block(cov_1, encoder_2, filters=filter_begin*4)  
decoder2 = decoder_block(decoder1, encoder_1, filters=filter_begin*2)  
decoder_out = decoder_block(decoder2, inputs, filters=filter_begin)
</code>
<br>  

 - model_2_96: 
<code>
filter_begin = 96  
encoder_1 = encoder_block(inputs, filters=filter_begin, strides=2)  
encoder_2 = encoder_block(encoder_1, filters=filter_begin*2, strides=2)  
encoder_out = encoder_block(encoder_2, filters=filter_begin*4, strides=2)  
cov_1 = conv2d_batchnorm(encoder_out, filters=filter_begin*4, kernel_size=1, strides=1)  
decoder1 = decoder_block(cov_1, encoder_2, filters=filter_begin*4)  
decoder2 = decoder_block(decoder1, encoder_1, filters=filter_begin*2)  
decoder_out = decoder_block(decoder2, inputs, filters=filter_begin)
</code>
<br> 

 - model_3_32: 
<code>
filter_begin = 32
encoder_1 = encoder_block(inputs, filters=filter_begin, strides=2)  
encoder_2 = encoder_block(encoder_1, filters=filter_begin*2, strides=2)  
encoder_3 = encoder_block(encoder_2, filters=filter_begin*4, strides=2)  
encoder_4 = encoder_block(encoder_3, filters=filter_begin*8, strides=2)  
encoder_out = encoder_block(encoder_4, filters=filter_begin*8, strides=2) 
cov_1 = conv2d_batchnorm(encoder_out, filters=filter_begin*8, kernel_size=1, strides=1)  
decoder1 = decoder_block(cov_1, encoder_4, filters=filter_begin*8)  
decoder2 = decoder_block(decoder1, encoder_3, filters=filter_begin*8)  
decoder3 = decoder_block(decoder2, encoder_2, filters=filter_begin*4)  
decoder4 = decoder_block(decoder3, encoder_1, filters=filter_begin*2)  
decoder_out = decoder_block(decoder4, inputs, filters=filter_begin)
</code>
<br> 
  
 - model_3_96:
<code>
filter_begin = 96
encoder_1 = encoder_block(inputs, filters=filter_begin, strides=2)  
encoder_2 = encoder_block(encoder_1, filters=filter_begin*2, strides=2)  
encoder_3 = encoder_block(encoder_2, filters=filter_begin*4, strides=2)  
encoder_4 = encoder_block(encoder_3, filters=filter_begin*8, strides=2)  
encoder_out = encoder_block(encoder_4, filters=filter_begin*8, strides=2) 
cov_1 = conv2d_batchnorm(encoder_out, filters=filter_begin*8, kernel_size=1, strides=1)  
decoder1 = decoder_block(cov_1, encoder_4, filters=filter_begin*8)  
decoder2 = decoder_block(decoder1, encoder_3, filters=filter_begin*8)  
decoder3 = decoder_block(decoder2, encoder_2, filters=filter_begin*4)  
decoder4 = decoder_block(decoder3, encoder_1, filters=filter_begin*2)  
decoder_out = decoder_block(decoder4, inputs, filters=filter_begin)
</code>
<br>


## Trainning Result
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


## Project Optimization

This encoder-decoder architecture FCN has can identify and locate objects in complex scenes. But if we wonder percept and trace other object(dog, cat, car, etc.), we also need more datas. The performance of the network may be good enough. However, the one we followed is wear red clothes, this will be easy to identify because no other one with red clothes, and the groud, sky, environment also has little red color. If we wonder our Quad-Copter get more power, we need more datas and more complex model, need more resourse and time to train and validation. Or we need other technology as like body gesture recognization.