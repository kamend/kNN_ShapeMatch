#pragma once

#include "ofMain.h"
#include "ofxCv.h"

class testApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();

		void keyPressed  (int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
	
	ofImage shape1;
	ofImage shape2;
	ofImage shape3;
	
	vector<cv::Point> points1;
	vector<cv::Point> points2;
	vector<cv::Point> points3;
	cv::KNearest *knn;
	
	bool drawMode;
	bool findNear;
	float found;
	vector<cv::Point> drawn;
};
