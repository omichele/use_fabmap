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
	if( argc != 2)
	{
		std::cout << "Problems" << std::endl;
		return -1;
	}


	// std::unordered_map<int, Mat> fabmapIDToKeyframe;
	std::vector<cv::Mat> memory;

	// creation of the modified FABMAP object -- with the method compare and add
	slam::FabMap fabMap;

	string input = argv[1];
//	string dirName = "'" + input + "'";

	string dirName = input;
	cout << "Directory chosen: " << dirName << endl;
	cout << "Directory chosen: " << dirName.c_str() << endl;



	DIR *dir;
	dir = opendir(dirName.c_str());
	struct dirent *ent;

	cv::Mat img;
	cv::Mat loop_img;

	ent = readdir(dir);

	cout << "Image to be read: " << ent->d_name << endl;

	ent = readdir(dir);

	cout << "Image to be read: " << ent->d_name << endl;

	ent = readdir(dir);

	cout << "Image to be read: " << ent->d_name << endl;

	ent = readdir(dir);

	cout << "Image to be read: " << ent->d_name << endl;

	if (dir != NULL) {
		if ( (ent = readdir(dir)) != NULL) {
			string imgPath = dirName + '/' + ent->d_name;
			img = imread(imgPath, CV_LOAD_IMAGE_COLOR);
			if(! img.data )                              // Check for invalid input
			{
				cout <<  "Could not open or find the image" << std::endl ;
				return -1;
			}
		}
	}

	int dstWidth = img.cols * 2;
	int dstHeight = img.rows * 2;

	cv::Mat dst = cv::Mat(dstHeight + 10, dstWidth + 10, CV_8UC3, cv::Scalar(70,70,70) );
	// cv::Rect roi(cv::Rect(0, 0, img.cols, img.rows));
	cv::Mat targetROI1 = dst( cv::Rect(0, 0, img.cols, img.rows) );
	cv::Mat targetROI2 = dst( cv::Rect(img.cols + 10 , 0, img.cols, img.rows) );
    cv::Mat targetROI3 = dst( cv::Rect(0 , img.rows + 10, img.cols, img.rows) );
	img.copyTo(targetROI1);
	img.copyTo(targetROI2);

	namedWindow("OpenCV Window");
	// show the image on window
	imshow("OpenCV Window", dst);
	// wait key for 5000 ms
	waitKey(40);
	//waitKey(0);

	// sleep(5000);



	if (! fabMap.isValid())
	{
		printf("Error: FabMap instance is not valid!\n");
		return -1;
	}


	if (dir != NULL) {
		while ( (ent = readdir(dir)) != NULL) {
			string imgPath = dirName + '/' + ent->d_name;
			Mat img = imread(imgPath, CV_LOAD_IMAGE_COLOR);
			if(! img.data )                              // Check for invalid input
			{
				cout <<  "Could not open or find the image" << std::endl ;
				return -1;
			}
			// cvtColor(img, img, CV_BGR2GRAY);



			int newID, loopID;
			fabMap.compareAndAdd(img, &newID, &loopID);
			if (newID < 0)
				return -1;

			// fabmapIDToKeyframe.insert(std::make_pair(newID, img));
			memory.push_back(img);
			if (loopID >= 0){
				cout << "LOOP CLOSURE!" << endl;
				cout << "Current image matches with image:" << loopID << "!!!!" << endl;
				cout << endl;
				cout << endl;
				cout << endl;
				// loop_img = fabmapIDToKeyframe.at(loopID);


				loop_img = memory[loopID];

//				namedWindow("test");
//
//				imshow("test", loop_img);

//				waitKey(0);

				loop_img.copyTo(targetROI3);
			}


			img.copyTo(targetROI1);
			img.copyTo(targetROI2);

			// show the image on window
			imshow("OpenCV Window", dst);



			waitKey(40);

			usleep(40);





		}
		closedir (dir);
	} else {
		cout << "not present" << endl;
	}


	// if(!useFabMap) return nullptr;


//	namedWindow("test");
//
//	imshow("test", img);




	return 0;


}


