# Transboost project
Boosting with transfer learning applied to image classification.  
__members__:  
- Luc Blassel
- Xue Bai
- Yifei Fan
- Romain Gautron
- Xuefan Xu

## chosen dataset
The CIFAR-10 dataset was chosen for it's simple structure, small images also lower computing times. It is also available in several ML libraries.

## chosen model
The Inception-V3 convolutional neural network was chosen. It is available, pre-trained, within a multitude of machine learning libraries.

## iterations
### first try (pure Tensor Flow)
In the _"fil rouge"_ folder.  
We managed to read data for only selected classes, and re-train inception to have a binary output.  However it got very complicated once we had to freeze certain layers and change structures of layers.  
So we decided to switch libraries

### second try (Tensor Flow with Keras front-end)
We decided to use Keras with TF back-end. It was a lot simpler to modify the network. We used the pre-trained version of Inception-V3 available in Keras, and added a binary output _softmax_ layer.  
We were also able to freeze layers and reinitialize weights in these to pave the way for boosting.

## TODO
- ~~Implement boosting algorithm~~
- ~~implement execution of script in GCloud environment~~
- Test boosting with weak projectors against a single strong projector
- Test boosting against specially trained model.
