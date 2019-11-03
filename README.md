# Self_Supervised_Autoencoders

fully-connected autoencoder in Pytorch for the CIFAR 10



# 1.Auto Encoder using Convolutions

I have used two convolution layers with filter size 5X5 in encoder, In the second convolution layer using stride=2 and padding to get the output of 16X16X3 i.e half of 32X32X3 for each channel.
In the decoder part two ConvTranspose2d  layers are used to bring back the original dimentions of the image back.

============== Encoder ==============\
Sequential(\
  (0): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))\
  (1): ReLU()\
  (2): Conv2d(6, 3, kernel_size=(5, 5), stride=(2, 2), padding=(4, 4))\
  (3): ReLU()\
)\
============== Decoder ==============\
Sequential(\
  (0): ConvTranspose2d(3, 6, kernel_size=(5, 5), stride=(2, 2), padding=(4, 4), output_padding=(1, 1))\
  (1): ReLU()\
  (2): ConvTranspose2d(6, 3, kernel_size=(5, 5), stride=(1, 1))\
  (3): Sigmoid()\
)\

Output: \
Top two rows the input \
bottom two rows corresponding output\

![CNN_AUTO](/uploads/0b3eb28675f8ea26f72ea436176c4419/CNN_AUTO.png)

# 2. Auto Encoder defined using Fully Connected Layers

I flattend the image (32X32X3) into one dimentional array to feed into fully connected layers(3072X1).
encoder with two fully connected layer reduced input to half (1536X1) and decoder with two fully connected layer again upsamples it back to original dimention.
============== Encoder ==============\
Sequential(\
  (0): Linear(in_features=3072, out_features=2304, bias=True)\
  (1): ReLU(inplace=True)\
  (2): Linear(in_features=2304, out_features=1536, bias=True)\
  (3): ReLU(inplace=True)\
)\
============== Decoder ==============\
Sequential(\
  (0): Linear(in_features=1536, out_features=2304, bias=True)\
  (1): ReLU(inplace=True)\
  (2): Linear(in_features=2304, out_features=3072, bias=True)\
  (3): Sigmoid()\
)\

Output: \
Top two rows the input \
bottom two rows corresponding output\
![FCN_AUTO](/uploads/43aa0a6c2416bce476105d2494a560e8/FCN_AUTO.png)

