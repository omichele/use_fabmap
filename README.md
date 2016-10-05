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


## Test

In order to launch the application run the executable in the /bin folder:

	./bin/use_fabmap_video_mono '/path/to/testSet' 'Image%05d.jpg' 1

The first argument is the path in which the image sequence is located. In fact in our case FABMAP was always executed offline when the
the acquisition of the test set was already done. Of course the program could be slightly modified to work online. The second argument
is the name format of the images to be read in the testSet folder.
