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


	// std::unordered_map<int, Mat> fabmapIDToKeyframe;
	std::vector<cv::Mat> memory;
	slam::FabMap fabMap;

	string input = argv[1];
	string imageName0 = argv[2];			// right camera normal lens
	int imageSkip = atoi(argv[3]);			// if the dataset is already sampled this is not needed
	

//	string dirName = "'" + input + "'";

	string dirName = input;
	cout << "Directory chosen: " << dirName << endl;
	cout << "Directory chosen: " << dirName.c_str() << endl;

	cv::VideoCapture sequence0(dirName + "/" + imageName0);

	// sequence0.set(CV_CAP_PROP_POS_MSEC, value);
	// sequence0.set(CV_CAP_PROP_POS_FRAMES, value);
	// sequence0.set(CV_CAP_PROP_FPS, 1);
	// sequence0.set(CV_CAP_PROP_FRAME_COUNT, value);
	// double value = sequence0.get(CV_CAP_PROP_FRAME_COUNT);


	cv::Mat img0;
	cv::Mat loop_img;


	sequence0 >> img0;

	// cout << "Image type: " << img0.type() << endl;

	// cvtColor(img, img, CV_BayerGR2RGB);

	int dstWidth = img0.cols * 2;
	int dstHeight = img0.rows * 2;

	cv::Mat dst = cv::Mat(dstHeight + 10, dstWidth + 10, img0.type(), cv::Scalar(70,70,70) );
	// cv::Rect roi(cv::Rect(0, 0, img.cols, img.rows));
	cv::Mat targetROI1 = dst( cv::Rect(0, 0, img0.cols, img0.rows) );
	cv::Mat targetROI2 = dst( cv::Rect(img0.cols + 10 , 0, img0.cols, img0.rows) );
    cv::Mat targetROI3 = dst( cv::Rect(0 , img0.rows + 10, img0.cols, img0.rows) );
    cv::Mat targetROI4 = dst( cv::Rect(img0.cols + 10 , img0.rows + 10, img0.cols, img0.rows) );
	img0.copyTo(targetROI1);
	// img1.copyTo(targetROI2);

	// prova text
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
	// waitKey(0);

	// sleep(5000);



	if (! fabMap.isValid())
	{
		printf("Error: FabMap instance is not valid!\n");
		return -1;
	}


	for(;;){
		
		// sequence0 >> img0;

		for( int ii=0; ii < imageSkip; ii++)
		{
			//cout << "imageSkip = " << imageSkip << endl;
		    //cout << "Skipping image num: " << ii << endl;
			sequence0 >> img0;
		}

		// prova text
		/*
		if (sequence0.get(CV_CAP_PROP_POS_FRAMES) + 10 <= sequence0.get(CV_CAP_PROP_FRAME_COUNT))
		{
			cout << "Total number of frames is: " << sequence0.get(CV_CAP_PROP_FRAME_COUNT) << endl;
			cout << "Next frame is: " << sequence0.get(CV_CAP_PROP_POS_FRAMES) << endl;
			// sequence0.set(CV_CAP_PROP_POS_FRAMES, sequence0.get(CV_CAP_PROP_POS_FRAMES) + 10);
			// sequence0.set(CV_CAP_PROP_POS_MSEC, sequence0.get(CV_CAP_PROP_POS_MSEC) + 100);
			// sequence0.set(CV_CAP_PROP_FPS, 10);      // not working
			cout << "Frames frequency is: " << sequence0.get(CV_CAP_PROP_FPS) << endl;
			cout << "Next frame is: " << sequence0.get(CV_CAP_PROP_POS_FRAMES) << endl;
		}
		else
			break;
	    */


		if(! img0.data )                              // Check for invalid inpu
		{
			cout <<  "Could not open or find the image" << std::endl ;
			return -1;
		}
		// cvtColor(img, img, CV_BGR2GRAY);

		int newID, loopID;
		double matchProb = fabMap.compareAndAdd(img0, &newID, &loopID, targetROI2);
		if (newID < 0)
			return -1;

		// fabmapIDToKeyframe.insert(std::make_pair(newID, img));
		memory.push_back(img0.clone());										// not good !!!!!!!!!!!!!


		if (loopID >= 0){
			cout << "LOOP CLOSURE!" << endl;
			cout << "Current image matches with image:" << loopID << "!!!!" << endl;
//			cout << endl;
//			cout << endl;
//			cout << endl;
			// loop_img = fabmapIDToKeyframe.at(loopID);


			loop_img = memory.at(loopID + 1);  // needed to make things work

//			double frameID = sequence0.get(CV_CAP_PROP_POS_FRAMES);
			// sequence0.set(CV_CAP_PROP_POS_FRAMES, (double)loopID + 2);

//			if (sequence0.set(CV_CAP_PROP_POS_FRAMES, (double)loopID + 2))
//				cout << "OK!" << endl;
//			else
//				cout << "Porco dio!!" << endl;
//
//			sequence0 >> loop_img;
//
//			sequence0.set(CV_CAP_PROP_POS_FRAMES, frameID);



			//				namedWindow("test");
			//
			//				imshow("test", loop_img);

			//				waitKey(0);

			loop_img.copyTo(targetROI3);

			sprintf(text1, "Prob %f", matchProb);
			cv::putText(text_img, text1, org, cv::FONT_HERSHEY_SIMPLEX, 3, Scalar( 255, 255, 255 ), 10);
			text_img.copyTo(targetROI4);
			text_img = cv::Mat(img0.rows, img0.cols, img0.type(), cv::Scalar(70,70,70));
		}


		img0.copyTo(targetROI1);

//		cv::Mat test = memory.front();
//		test.copyTo(targetROI4);
//		cv::Mat test;
//		if(memory.size() > 16){
//			cout << "Baobab" << endl;
//			test = memory.at(16);
//			test.copyTo(targetROI4);
//		}
		cv::Mat image_roi = targetROI2;

		// show the image on window
		imshow("OpenCV Window", dst);
		imshow("Features", image_roi);
		waitKey(1);


		if (loopID >= 0){
			sleep(1);
		}

		// usleep(500);





	}



	// if(!useFabMap) return nullptr;


	//	namedWindow("test");
	//
	//	imshow("test", img);




	return 0;


}




