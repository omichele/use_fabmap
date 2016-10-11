#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
// connection to openFABMAP
#include "openfabmap.hpp"


namespace of2 {
	class FabMap;
}

namespace cv {
	class FeatureDetector;
	class BOWImgDescriptorExtractor;
}

namespace slam {

/** Interface to openFabMap. */
class FabMap
{
public:
	/** Initializes FabMap. */
	FabMap();
	
	/** Writes out the confusion matrix if enabled. */
	~FabMap();
	
	/** Adds the keyframe to the set of frames to compare against and returns
	 *  its (non-negative) ID in FabMap (different to the keyframe ID).
	 *  Returns -1 if the frame cannot be added due to an error. */
// 	int add(KeyFrame* keyframe);
	
	/** Checks if the keyframe is determined to be the same as an already
	 *  added frame and if yes, returns its ID. If not, returns -1.
	 *  Does not directly return a KeyFrame pointer to allow for KeyFrames
	 *  being deleted. */
// 	int compare(KeyFrame* keyframe);

	/** Combination of compare() followed by add() (more efficient). */
	double compareAndAdd(cv::Mat keyframe, int* out_newID, int* out_loopID, cv::Mat targetROI);
	
	void compareAndAdd(cv::Mat keyframe, int* out_newID, int* out_loopID);

	void drawRichKeypoints(const cv::Mat& src, std::vector<cv::KeyPoint>& kpts, cv::Mat& dst);

	/** Returns if the class is initialized correctly (i.e. if the required
	 *  files could be loaded). */
	bool isValid() const;
	
private:
	int nextImageID;
	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::BOWImgDescriptorExtractor> bide;

	// instance of the original openFABMAP
	cv::Ptr<of2::FabMap> fabMap;
	
	// enable the construction of the confusion matrix
	const bool printConfusionMatrix = true;
	cv::Mat confusionMat;
	
	bool valid;

	const float minLoopProbability = 0.80f;
	// this tolerance represent the number of frames
	const int tolerance = 40;

};

}

