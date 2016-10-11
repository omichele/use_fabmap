#include "FabMap.h"
#include <dirent.h>
#include <unordered_map>
#include <utility>
#include <unistd.h>

using namespace std;
using namespace slam;
using namespace cv;



int main(int argc, char* argv[])
{
	if( argc != 4)
	{
		std::cout << "Problems" << std::endl;
		return -1;
	}


	std::vector<cv::Mat> memory;
	// create instance of the fabmap class
	slam::FabMap fabMap;

	string input = argv[1];                 // test set path
	string imageName0 = argv[2];			// right camera normal lens
	int imageSkip = atoi(argv[3]);			// if the dataset is already sampled this is not needed
	

	string dirName = input;
	cout << "Directory chosen: " << dirName << endl;
	cout << "Directory chosen: " << dirName.c_str() << endl;

	cv::VideoCapture sequence0(dirName + "/" + imageName0);

	cv::Mat img0;  		// it will contain the current image read
	cv::Mat loop_img; 	// it will contain the matched image 

	// read the image
	sequence0 >> img0;

	int dstWidth = img0.cols * 2;
	int dstHeight = img0.rows * 2;
    
    // dst will be the image visualized in our GUI
	cv::Mat dst = cv::Mat(dstHeight + 10, dstWidth + 10, img0.type(), cv::Scalar(70,70,70) );
	// Set up four different regions in our GUI
	// ROI1 = original left image - top left corner
	cv::Mat targetROI1 = dst( cv::Rect(0, 0, img0.cols, img0.rows) );
	// ROI2 = original right image (or empty if there is none) - top right corner
	cv::Mat targetROI2 = dst( cv::Rect(img0.cols + 10 , 0, img0.cols, img0.rows) );
	// ROI3 = matching image - bottom right corner
    cv::Mat targetROI3 = dst( cv::Rect(0 , img0.rows + 10, img0.cols, img0.rows) );
    // ROI4 plot the matching probability when a detection occurs
    cv::Mat targetROI4 = dst( cv::Rect(img0.cols + 10 , img0.rows + 10, img0.cols, img0.rows) );
    
	img0.copyTo(targetROI1);
	// img1.copyTo(targetROI2);

	cv::Point org;
	org.x = img0.cols/2 - 400;
	org.y = img0.rows/2;
	char text0[255];
	sprintf(text0, "Prob:");
	char text1[255];


	// disply it in targetROI4
	cv::Mat text_img = cv::Mat(img0.rows, img0.cols, img0.type(), cv::Scalar(70,70,70));
	cv::putText(text_img, text0, org, cv::FONT_HERSHEY_SIMPLEX, 3, Scalar( 255, 255, 255 ), 10);

	text_img.copyTo(targetROI4);
	text_img = cv::Mat(img0.rows, img0.cols, img0.type(), cv::Scalar(70,70,70));


	namedWindow("OpenCV Window", cv::WINDOW_NORMAL);
	namedWindow("Features", cv::WINDOW_NORMAL);
	// show the image on window
	imshow("OpenCV Window", dst);
	// wait key for 5000 ms
	waitKey(40);

	if (! fabMap.isValid())
	{
		printf("Error: FabMap instance is not valid!\n");
		return -1;
	}

	for(;;){

		for( int ii=0; ii < imageSkip; ii++)
		{
			//cout << "imageSkip = " << imageSkip << endl;
		    //cout << "Skipping image num: " << ii << endl;
			sequence0 >> img0;
		}

		if(! img0.data )   // Check for invalid input
		{
			cout <<  "Could not open or find the image" << std::endl ;
			return -1;
		}


		int newID, loopID;
		// This is a call to fabmap that is the core of the algorithm
		// It returns the matching probability and it fills newID and 
		// loopID
		double matchProb = fabMap.compareAndAdd(img0, &newID, &loopID, targetROI2);
		if (newID < 0) return -1;

		memory.push_back(img0.clone());

		if (loopID >= 0){
			cout << "LOOP CLOSURE!" << endl;
			cout << "Current image matches with image:" << loopID << "!!!!" << endl;

			loop_img = memory.at(loopID + 1);  // + 1 is needed to make things work

            // the matching image is displayed in ROI3
			loop_img.copyTo(targetROI3);

			sprintf(text1, "Prob %f", matchProb);
			cv::putText(text_img, text1, org, cv::FONT_HERSHEY_SIMPLEX, 3, Scalar( 255, 255, 255 ), 10);
			text_img.copyTo(targetROI4);
			text_img = cv::Mat(img0.rows, img0.cols, img0.type(), cv::Scalar(70,70,70));
		}


		img0.copyTo(targetROI1);
		// We want to plot just the features detected in the current image
		cv::Mat image_roi = targetROI2;

		// show the image on window
		imshow("OpenCV Window", dst);
		imshow("Features", image_roi);
		waitKey(1);

		// freeze the screen for a while at every match
		if (loopID >= 0){
			sleep(1);
		}

	}
	return 0;

}




