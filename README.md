# Finding and Tracking Vehicles 

<img src="Examples/HOG Example.jpg" width="300" alt="Combined Image" /> <img src="Examples/Processing Example.jpg" width="275" alt="Combined Image" />
<img src="Examples/Output Example.jpg" width="300" alt="Combined Image" />


Skills
---
* Computer Vision
* Machine Learning 
* Python


Overview
---
A Support Vector Machine is trained for vehicle identification using a feature vector containing HOG (Histogram of oriented gradients) features (extracted from HLS color space), as well as spacial color and color histogram data. The SVM is then applied to an example video, and vehicles are tracked using a heatmap approach across frames. 

This project was completed as an assignment for Udacity's Self Driving Car Nano Degree [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive) - no starter code was used in the creation of this project. 


Dependencies
---
This project was built using the anaconda _carnd-term1_ environment available [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md)


Contents
---
* project_video_output.mp4
    * final output video with vehicles identified and tracked
* feature_extraction.py
    * python script providing functions for extraction of HOG, spacial color, and color histogram data from images
* vehicles
    * training images (of vehicles) used for the SVM, were taken from [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) 
* non-vehicles
    * training images (of non-vehicles) used for the SVM, were taken from [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)
* train_classifier.ipynb
    * jupyter notebook containing code to execute training (and saving) of SVM. 
* svc.pkl
    * pickle file containing trained Support Vector Machine (classifier)
* X_scaler.pkl
    * pickle file containing scaling parameters for uniformly scaling feature vectors before running through SVM
* features_scan.py
    * python script with functions for using SVM to predict whether a provided image contains a vehicle
* id_label.py
    * python script providing functions for tracking identified vehicles via heat map
* processing_pipeline.ipynb
    * jupyter notebook containing code to execute analysis of video stream for vehicle identification and tracking
* p5 writeup.pdf
    * additional thoughts from project creation


