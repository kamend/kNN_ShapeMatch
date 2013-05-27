#include "testApp.h"


void getOutterContour(ofImage &im, vector<cv::Point> &points) {
	
	cv::Mat original;
	original = ofxCv::toCv(im).clone();
	
	// 1-channel convert
	if(original.channels() == 3) {
		cv::cvtColor(original, original, CV_RGB2GRAY);
	} else if(original.channels() == 4) {
		cv::cvtColor(original, original, CV_RGBA2GRAY);
	}
	
	// Canny
	cv::Canny(original, original, 0, 100.0);
	
	cv::Mat dilateKernel(cv::Size(3,3), CV_8UC1, cv::Scalar(1));
	cv::dilate(original, original, dilateKernel);
	
	vector<vector<cv::Point> > foundc;
	cv::findContours(original, foundc,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point(0,0));
	if(foundc.size() >0) {
		points = foundc[0];
	} else {
		ofLog() << "No shapes found!";
	}
	

}


//--------------------------------------------------------------
void testApp::setup(){
		drawMode = true;
	ofSetFrameRate(30);
	
	shape1.loadImage("circle.png");
	shape2.loadImage("rect.png");
	shape3.loadImage("triangle_shifted.png");
	
	getOutterContour(shape1, points1);
	getOutterContour(shape2, points2);
	getOutterContour(shape3, points3);
	
	
	
	int numPoints = points1.size()+points2.size()+points3.size();

	// training data
	cv::Mat trainingClasses(numPoints, 1, CV_32FC1);
	cv::Mat trainingData(numPoints, 2, CV_32FC1);

	for(int i=0;i<points1.size();i++) {

		trainingData.at<float>(i,0) = points1[i].x;
		trainingData.at<float>(i,1) = points1[i].y;
		trainingClasses.at<float>(i,0) = 1;
	}
	
	for(int i=points1.size();i<points1.size()+points2.size();i++) {

	
		int realIndex = i-points1.size();
		trainingData.at<float>(i,0) = points2[realIndex].x;
		trainingData.at<float>(i,1) = points2[realIndex].y;
		trainingClasses.at<float>(i,0) = 2;
	}
	ofLog() << points3.size();
	
	for(int i=points1.size()+points2.size();i<numPoints;i++) {
		int realIndex = i - (points1.size()+points2.size());
		
		trainingData.at<float>(i,0) = points3[realIndex].x;
		trainingData.at<float>(i,1) = points3[realIndex].y;

		trainingClasses.at<float>(i,0) = 3;
	}
	
	knn = new cv::KNearest(trainingData, trainingClasses,cv::Mat(), false,1);
	
	findNear = false;
	found = -1;
}

//--------------------------------------------------------------
void testApp::update(){
	if(findNear) {
		if(drawn.size()>1) {
			
			cv::Mat testData(drawn.size(), 2, CV_32FC1);
			cv::Mat predicted(testData.rows, 1, CV_32FC1);
			
			for(int i=0;i<drawn.size();i++) {
				testData.at<float>(i,0) = drawn[i].x;
				testData.at<float>(i,1) = drawn[i].y;
			}
			found = knn->find_nearest(testData, 1);
			
		}
		findNear = false;
			
	}
}

//--------------------------------------------------------------
void testApp::draw(){
	ofBackground(0, 0, 0);
	
	ofSetColor(255,255,255);
	
	ofNoFill();
	
	ofRect(0,0,200,200);
	
	if(found == 1) {
		shape1.draw(0,0);
	} else if(found == 2) {
		shape2.draw(0,0);
	} else if(found == 3) {
		shape3.draw(0, 0);
	}
	
	ofSetColor(255,0,0);
	for(int i=0;i<drawn.size();i++) {
		ofCircle(drawn[i].x, drawn[i].y,2);
	}
	
	ofSetColor(0,0,255);
	for(int i=0;i<points1.size();i++) {
		ofCircle(points1[i].x, points1[i].y, 1);
	}
	for(int i=0;i<points2.size();i++) {
		ofCircle(points2[i].x, points2[i].y, 1);
	}
	
	for(int i=0;i<points3.size();i++) {
		ofCircle(points3[i].x, points3[i].y, 1);
	}
	
	
	
}

//--------------------------------------------------------------
void testApp::keyPressed(int key){
	if(key == 'd') {
		drawMode = !drawMode;
	}

}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){
	
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){
	if(drawMode) {
		
		cv::Point p = cv::Point(x,y);
	
		drawn.push_back(p);
	}
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
	if(drawMode) {
		found = -1;
		drawn.clear();
	}
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
	if(drawMode) {
		findNear = true;
	}
}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 

}