# use_fabmap

This is a software packet that implements openFABMAP software [1] and provides a GUI for monitoring and assessing the recognition.
OpenFABMAP in turn is the open source implementation of the original FABMAP software presented in [2,3]. Some parts of the code were
taken from the software package lsd_slam ( https://github.com/tum-vision/lsd_slam ) which makes uses of openFABMAP as a complementary
module for visual place recognition (loop closure events in SLAM). Refer to the mentionend papers for accurate explaination of FABMAP
and its open source implementation.


## Requirements

openFabMap (internal)
OpenMP
OpenCV

## Installation

Run 
	cmake .
	make

## Parameters

In the file FabMap.h there is a couple of hard coded parameters very importent to be kept into account for the recognition namely 
minLoopProbability and tolerance. The latter represent the threshold to be applied to the matching probability in order to deem it
as an actual match. The original value was set to 0.99 but we used also 0.80. This parameter in fact is in our case just functional
to the visualization. A sweeping of the threshold values on the confusion matrix can always be performed later in order to quantatively
assess the performances. The former parameter represent the number of recent frames which are not considered for the search of a match.
This value can be adjusted according to the frame rate of the images, the type of environment and the velocity of the camera during its
path. Again this is just for the visualization purpose and another tolerance can be set on the confusion matrix at a later time.


One other important setting for the software also in FabMap.cpp is:

	const bool printConfusionMatrix = true;

Once the printing of the confusion matrix is enabled the software will produce at the end of its run a txt file containing the values 
of this matrix that are basically the matching scores indexed by the current images (rows) and the memory images (columns). An accurate
explaination of the importance of the confusion matrix for the performance analysis is given in [].

The file FabMap.cpp contains the implementation of the class FabMap that is used then in the main program. In the constructor there are
some hard coded params specifying the paths of the vocabulary, the chow-liu tree, the training data and the result folder. Please refer
to the documentation of openFabMap for the training part.


## Test

In order to launch the application run the executable in the /bin folder:

	./bin/use_fabmap_video_mono '/path/to/testSet' 'Image%05d.jpg' 1

The first argument is the path in which the image sequence is located. In fact in our case FABMAP was always executed offline when the
the acquisition of the test set was already done. Of course the program could be slightly modified to work online. The second argument
is the name format of the images to be read in the testSet folder. The third argument is a number used for a fixed rate subsampling of
the test set, specifying the number of images to be discarded before retaining the next. If the test set was already subsampled (like 
in our experiments) this number can be set to 1.


