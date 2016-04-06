#include "FabMap.h"
#include <dirent.h>
#include <unordered_map>
#include <utility>
#include <unistd.h>

using namespace std;
using namespace slam;
using namespace cv;



int main(int argc, char * argv[])
{
	if( argc != 3)
	{
		std::cout << "Problems" << std::endl;
		return -1;
	}


	// std::unordered_map<int, Mat> fabmapIDToKeyframe;
	std::vector<cv::Mat> memory;
	slam::FabMap fabMap;

	string input = argv[1];
	string imageName0 = argv[2];			// left camera
	string imageName1 = argv[3];			// right camera

//	string dirName = "'" + input + "'";

	string dirName = input;
	cout << "Directory chosen: " << dirName << endl;
	cout << "Directory chosen: " << dirName.c_str() << endl;

	cv::VideoCapture sequence0(dirName + "/" + imageName0);

	// cv::VideoCapture sequence1(dirName + "/" + imageName1);

	cv::VideoCapture sequence1;

	if( argc == 3){
		string imageName1 = argv[3];
		sequence1.open(dirName + "/" + imageName1);
	}
	else
		sequence1.open(dirName + "/" + imageName0);


	cv::Mat img0;
	cv::Mat img1;
//	cv::Mat deb;
	cv::Mat loop_img;


	sequence0 >> img0;

	sequence1 >> img1;

	// cout << "Image type: " << img0.type() << endl;

	// cvtColor(img, img, CV_BayerGR2RGB);
	// deb = img;

	int dstWidth = img0.cols * 2;
	int dstHeight = img0.rows * 2;

	cv::Mat dst = cv::Mat(dstHeight + 10, dstWidth + 10, img0.type(), cv::Scalar(70,70,70) );
	// cv::Rect roi(cv::Rect(0, 0, img.cols, img.rows));
	cv::Mat targetROI1 = dst( cv::Rect(0, 0, img0.cols, img0.rows) );
	cv::Mat targetROI2 = dst( cv::Rect(img0.cols + 10 , 0, img0.cols, img0.rows) );
    cv::Mat targetROI3 = dst( cv::Rect(0 , img0.rows + 10, img0.cols, img0.rows) );
    cv::Mat targetROI4 = dst( cv::Rect(img0.cols + 10 , img0.rows + 10, img0.cols, img0.rows) );
	img0.copyTo(targetROI1);
	img1.copyTo(targetROI2);

	namedWindow("OpenCV Window");
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
		sequence0 >> img0;

		sequence1 >> img1;


		if(! img0.data )                              // Check for invalid input
		{
			cout <<  "Could not open or find the image" << std::endl ;
			return -1;
		}
		// cvtColor(img, img, CV_BGR2GRAY);

		if(! img1.data )                              // Check for invalid input
		{
			cout <<  "Could not open or find the image" << std::endl ;
			return -1;
		}



		int newID, loopID;
		fabMap.compareAndAdd(img0, &newID, &loopID, targetROI4);
		if (newID < 0)
			return -1;

		// fabmapIDToKeyframe.insert(std::make_pair(newID, img));
		memory.push_back(img0.clone());

		if (loopID >= 0){
			cout << "LOOP CLOSURE!" << endl;
			cout << "Current image matches with image:" << loopID << "!!!!" << endl;
//			cout << endl;
//			cout << endl;
//			cout << endl;
			// loop_img = fabmapIDToKeyframe.at(loopID);


			loop_img = memory.at(loopID + 2);  // needed to make things work

			//				namedWindow("test");
			//
			//				imshow("test", loop_img);

			//				waitKey(0);

			loop_img.copyTo(targetROI3);
		}


		img0.copyTo(targetROI1);
		img1.copyTo(targetROI2);

//		cv::Mat test = memory.front();
//		test.copyTo(targetROI4);
//		cv::Mat test;
//		if(memory.size() > 16){
//			cout << "Baobab" << endl;
//			test = memory.at(16);
//			test.copyTo(targetROI4);
//		}

		// show the image on window
		imshow("OpenCV Window", dst);



		waitKey(40);

		// usleep(500);





	}



	// if(!useFabMap) return nullptr;


	//	namedWindow("test");
	//
	//	imshow("test", img);




	return 0;


}




