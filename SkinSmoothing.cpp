#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include "dataPath.hpp"

using namespace std;
using namespace cv;

/** Global variables */
String faceCascadePath;
CascadeClassifier faceCascade;

double getHueMean(Mat image) {
    cvtColor(image,image,COLOR_BGR2HSV);
    Mat imageChannels[3];
    split(image,imageChannels);
    Scalar mean,dev;
    meanStdDev(imageChannels[0], mean, dev);
    return mean.val[0];
}

void getLowerUpperVal(int mean,int& lower,int& upper)
{
	lower = mean -10;

	if(lower < 0)
		lower = 0;

	upper = mean + 10;

	if(upper > 255)
		upper = 255;
}

int main( int argc, const char** argv )
{
	int faceNeighborsMax = 10;
	int neighborStep = 1;
	faceCascadePath = DATA_PATH + "models/haarcascade_frontalface_default.xml";

	if( !faceCascade.load( faceCascadePath))
	{
	  cout<<"Error loading face cascade"<<endl;
	  return -1;
	}

	// Read image
	Mat frame = imread(DATA_PATH + "images/hillary_clinton.jpg");

	Mat frameGray;
	cvtColor(frame, frameGray, COLOR_BGR2GRAY);

	std::vector<Rect> faces;
	Mat frameClone = frame.clone();
	int neighbours = 10;
	faceCascade.detectMultiScale( frameGray, faces, 1.2, neighbours);

	Mat face = frame.clone();

	for(int i = 0; i < faces.size();i++)
	{
		int x = faces[i].x;
		int y = faces[i].y;
		int w = faces[i].width;
		int h = faces[i].height;

		rectangle(face, Point(x, y), Point(x + w, y + h), Scalar(255,0,0), 2, 4);

	  //Now get the skin patch
	  //The detected face region will have the chick on the left and right side of the mid point of height
	  //So we can select a patch from the chick region

		int patchX = x + faces[i].width/2 + faces[i].width/8;
		int patchY = y + faces[i].height/2;
		int patchWidth = faces[i].width/8;
		int patchHeight = faces[i].height/8;

		Mat skinPatch = frameClone(Range(patchY,patchY+patchHeight),Range(patchX,patchX + patchWidth));

		double hueMean = getHueMean(skinPatch);

		int hlower,hupper;

		getLowerUpperVal(hueMean,hlower,hupper);

		Mat hsv;
		cvtColor(frame,hsv,COLOR_BGR2HSV);
		Mat skinMask;
		inRange(hsv,Scalar(hlower,0,0),Scalar(hupper,255,255),skinMask);

		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
		erode(skinMask, skinMask,kernel,Point(-1,-1),2);
		dilate(skinMask,skinMask, kernel,Point(-1,-1),2);

		GaussianBlur(skinMask,skinMask, Size(3, 3), 0);

		Mat inverseMask;
		threshold(skinMask,inverseMask,0,255,THRESH_BINARY_INV);

		Mat skin,nonSkin,bblur;
		bitwise_and(frame,frame,skin,skinMask);
		bitwise_and(frame,frame,nonSkin,inverseMask);

		bilateralFilter(skin,bblur,9,75,75);

		imshow("Input",frame);
		imshow("Face",face);

		Mat output;
		bitwise_or(nonSkin,bblur,output);

		imshow("Output",output);
	}

	waitKey(0);

	return 0;
}
