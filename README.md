# PlasticDetectionYOLO

This is from ECOLYolo's Fall 2020 Senior Design project. 


# Introduction

This user guide covers how to set up and use darknet for plastic detection as well as training the model to create the necessary weights. To use the pre-trained weights, only the Using Detection Model is needed. 

This guide utilized Python 3.6, OpenCV 4.5.0, CUDA, cuDNN as the main dependencies and requirements for using and training the model. Make sure to install these before attempting to install further dependencies. All installation instructions and command line prompts will assume a Unix OS for the coding environment. 

This project uses darknet, a neural network designed for YOLO created by Joseph Redmon and adapted. Additional information about YOLO and its different versions can be found at [1].  The OpenCV installation was done by building from source due to some complications with pip by following the guide given at [2]. If pip is available, use that method for installation of OpenCV. The sources for the database used and darknet are found at [3] and [4] respectively. 



# Preparing the Dependencies

Installing the dependencies is necessary in order to complete training and processing within reasonable time constraints. 

For CUDA and cuDNN, look up the graphics card used and refer to the chart in [5] to install the correct version of CUDA and check NVIDIA’s website to install the correct version of cuDNN which is based upon the CUDA version. 

To install OpenCV, attempt to use the pip instruction found below; however, if that does not work. Search for “install opencv [operating system] from source” in order to build it from source. 

Furthermore, to improve threading and speed, also install OpenMP by searching “install OpenMP [operating system]. 

Using

`$pip install [package-name]`

install the following packages below: 
Cython
Numpy
Matplotlib
Scipy
opencv-python *note:(if the correct package isn’t installed, specify the version by doing opencv-python==[verison.number.here]
awscli



# 3. Train the Model

To train new weights for the model, darknet must be installed. To install from source, complete follow the directions found at website [1]. 

To adjust darknet’s capabilities, change the makefile (Makefile) in the darknet folder setting GPU=1 if there is an NVIDIA GPU available on the used machine. Additionally, if cuDNN is installed, change CUDNN=1. Set OpenCV=1. Also, if OpenMP is installed, set OpenMP=1. These settings and modules installed will make the training process much faster. 

To build the package run
`$make`

In the Trash-ICRA19 folder in cfgs_and_ckpts/yolo/cfg, adjust the following files to have the correct working directory paths: 
test.txt
train.txt
valid.txt
yolo.data

To begin training, change the directory back to the darknet folder and run 

`$./darknet detector train /full_path_to/yolo.data /full_path_to/in_trash-icra19/yolo.cfg ./desired_beginning_weights &> /path_to/train.log`



# 4. Using the Detection Model

The different ways in using the detection model in darknet can be found on the READ.me linked in [4]. It is possible to apply the detection model to IP cameras, images, and videos. 



# Miscellaneous Notes

The format of the user manual was taken from [5] from a previous project. 

In the event there is a desire to use the most recent version of YOLO. Make the following changes in MakeFile in darknet. 

Change the section

`ifeq ($(OPENCV), 1)
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` -lstdc++
COMMON+= `pkg-config --cflags opencv`
endif`

to

`ifeq ($(OPENCV), 1)
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv4` -lstdc++
COMMON+= `pkg-config --cflags opencv4`
endif`

Additionally, in ./src/image_opencv.cpp remove the following lines: 
`IplImage *image_to_ipl(image im)
{
   int x,y,c;
   IplImage *disp = cvCreateImage(cvSize(im.w,im.h), IPL_DEPTH_8U, im.c);
   int step = disp->widthStep;
   for(y = 0; y < im.h; ++y){
       for(x = 0; x < im.w; ++x){
           for(c= 0; c < im.c; ++c){
               float val = im.data[c*im.h*im.w + y*im.w + x];
               disp->imageData[y*step + x*im.c + c] = (unsigned char)(val*255);
           }
       }
   }
   return disp;
}
image ipl_to_image(IplImage* src)
{
   int h = src->height;
   int w = src->width;
   int c = src->nChannels;
   image im = make_image(w, h, c);
   unsigned char *data = (unsigned char *)src->imageData;
   int step = src->widthStep;
   int i, j, k;
   for(i = 0; i < h; ++i){
       for(k= 0; k < c; ++k){
           for(j = 0; j < w; ++j){
               im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
           }
       }
   }
   return im;
}
Mat image_to_mat(image im)
{
   image copy = copy_image(im);
   constrain_image(copy);
   if(im.c == 3) rgbgr_image(copy);
   IplImage *ipl = image_to_ipl(copy);
   Mat m = cvarrToMat(ipl, true);
   cvReleaseImage(&ipl);
   free_image(copy);
   return m;
}
image mat_to_image(Mat m)
{
   IplImage ipl = m;
   image im = ipl_to_image(&ipl);
   rgbgr_image(im);
   return im;
}`

Add the following lines in ./src/image_opencv.cpp: 

`Mat image_to_mat(image im)
{
image copy = copy_image(im);
constrain_image(copy);
if(im.c == 3) rgbgr_image(copy);
Mat m(cv::Size(im.w,im.h), CV_8UC(im.c));
int x,y,c;
int step = m.step;
for(y = 0; y < im.h; ++y){
   for(x = 0; x < im.w; ++x){
       for(c= 0; c < im.c; ++c){
           float val = im.data[c*im.h*im.w + y*im.w + x];
           m.data[y*step + x*im.c + c] = (unsigned char)(val*255);
       }
   }
}
free_image(copy);
return m;
}
image mat_to_image(Mat m)
{
int h = m.rows;
int w = m.cols;
int c = m.channels();
image im = make_image(w, h, c);
unsigned char *data = (unsigned char *)m.data;
int step = m.step;
int i, j, k;
for(i = 0; i < h; ++i){
   for(k= 0; k < c; ++k){
       for(j = 0; j < w; ++j){
           im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
       }
   }
}
rgbgr_image(im);
return im;
}`


References
[1] J. Redmon, “Darknet: Open Source Neural Networsk in C”. pjreddie.com. 2013-2020. [Online]. Available: https://pjreddie.com/darknet/. 
[2] “Build and install OpenCV from source”. RIP Tutorial. [Online]. Available: https://riptutorial.com/opencv/example/15781/build-and-install-opencv-from-source. 
[3] Interactive Robotics and Vision Lab, “Trash-ICRA19,” [Online]. Available: http://irvlab.cs.umn.edu/resources/trash-icra19 [Accessed Sept. 20, 2020].
[4] AlexeyAB (2017) darknet [Source Code]. https://github.com/AlexeyAB/darknet/tree/cad4d1618fee74471d335314cb77070fee951a42.
[5] “Urchin Recongition Guide.”(2019). 




