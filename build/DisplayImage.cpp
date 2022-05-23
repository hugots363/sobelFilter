#include <stdio.h>
#include <string>
#include <cmath>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <experimental/filesystem>
namespace filesys = std::experimental::filesystem;

//Include for openmp parallelization
#include <omp.h>

//Namespaces
using namespace cv;
using namespace std;


//Global vars
const int kerx[9]= {-1,0,1,-2,0,2,-1,0,1};
const int kery[9] = {1,2,1,0,0,0,-1,-2,-1};


string getImgPath(){
  std::cout << "Write the absolute path of the image:" << '\n';
  std::string path;
  std::getline(std::cin, path);
  return path;
}

string getDumpPath(){
  std::cout << "Write the path for the new image:" << '\n';
  std::string path;
  std::getline(std::cin, path);
  return path;
}

void saveImage(string path, Mat image){
  //imwrite(dump_path,image );
  imwrite(path,image );
}

void displayImage(Mat image){
  String windowName = "Sobel";
  namedWindow(windowName); // Create a window
  imshow(windowName, image); // Show our image inside the created window.
  waitKey(0); // Wait for any keystroke in the window
  destroyWindow(windowName); //destroy the created window
}


//Return of the part of the pixel applying kernel x
int pixelX(int i0,int i1,int i2, int i3, int i4, int i5, int i6,int i7,int i8){
  int res = i0*kerx[0]+ i1*kerx[1]+ i2*kerx[2]+
            i3*kerx[3]+ i4*kerx[4]+ i5*kerx[5]+
            i6*kerx[6]+ i7*kerx[7]+ i8*kerx[8];

  return res;
}

//Return of the part of the pixel applying kernel y
int pixelY(int i0,int i1,int i2, int i3, int i4, int i5, int i6,int i7,int i8){
  int res = i0*kery[0]+ i1*kery[1]+ i2*kery[2]+
            i3*kery[3]+ i4*kery[4]+ i5*kery[5]+
            i6*kery[6]+ i7*kery[7]+ i8*kery[8];

  return res;
}

//Just pythagoras theorem and assuring pixel limits(0 to 255)
int pythagoras(int a, int b){
  int res = sqrt((a*a)+(b*b));
  if (res > 255) {return 255;}
  if (res < 0) {return 0;}
  return res;
}

//Iteration
Mat sobelFilter(Mat image, Mat resImg){

  //Iteration through matrix pixels and declaration of private variables
  int r;
  int c;
  int px;
  int py;
  #pragma omp parallel for num_threads(4) private(r,px,py)
  for(c = 0; c < image.cols ; c++){
    for( r = 0; r < image.rows ; r++){
      if(r == 0 || c == 0 || r == image.rows -1 || c== image.cols -1){
        resImg.at<uchar>(r,c) = 255;
      }
      else{
        px =  pixelX(image.at<uchar>(r-1,c-1), image.at<uchar>(r-1,c),image.at<uchar>(r-1,c+1),
                        image.at<uchar>(r,c-1),image.at<uchar>(r,c),image.at<uchar>(r,c+1),
                        image.at<uchar>(r+1,c-1),image.at<uchar>(r+1,c),image.at<uchar>(r+1,c+1));

        py = pixelY(image.at<uchar>(r-1,c-1), image.at<uchar>(r-1,c),image.at<uchar>(r-1,c+1),
                        image.at<uchar>(r,c-1),image.at<uchar>(r,c),image.at<uchar>(r,c+1),
                        image.at<uchar>(r+1,c-1),image.at<uchar>(r+1,c),image.at<uchar>(r+1,c+1));
        resImg.at<uchar>(r,c) = pythagoras(px,py);
      }
    }
  }
  return resImg;
}



int main(void)
{
  std::cout << "Program to apply Sobel filter into an image" << "\n";
  //Get path of the image to work with
  std::string img_path = getImgPath();

  Mat image = imread(img_path, IMREAD_GRAYSCALE);
  // Check for failure
  if (image.empty())
  {
   cout << "Could not open or find the image" << endl;
   cin.get();
   return -1;
  }

  //Initializing the Mat to store the result
  Mat resImg = Mat::zeros(image.size(), CV_8UC1);

  //apply Sobel filter and saving the result in resImg
  sobelFilter(image, resImg);

  //Getting the path for the new image with Sobel applied
  string dump_path =  getDumpPath();
  //Saving the image in the same path with _sobel sufix
  saveImage(dump_path, resImg);

  //In case an inversion of colour is needed
  /*
  Mat aux;
  bitwise_not(resImg,aux);
  displayImage(aux);
  */

  //Displaying the result
  displayImage(resImg);


  return 0;
}
