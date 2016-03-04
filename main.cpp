#include <stdlib.h>
#include <cv.hpp>
#include <cxcore.hpp>
#include <highgui.h>

#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;


void print_var(string hint,cv::Mat img){
cout << hint << endl << " "  << img << endl << endl;
}

void print_var(string hint,float float_num){
cout << hint << endl << " "  << float_num << endl << endl;
}

// Calculates the Gaussian Blur and stores the result on the height map given
cv::Mat GaussianBlur(cv::Mat img,int gaussian_kernel_size,float segma)
{

    if(gaussian_kernel_size%2 == 0)
    {
        cout << "Use odd size for gaussian kernel ";
        return img;
    }

    if(img.cols!=img.rows){
        cout << "Use Image with same width and height";
        return img;
    }


    if(img.cols%2 != 0){
        cout << "Use Image with size 2 power n";
        return img;
    }

    // init the 1D gaussian Kernel using open CV
    cv::Mat gussian_kernal=cv::getGaussianKernel(gaussian_kernel_size,segma);
    print_var("g_ker",gussian_kernal);

    //init the variables
    cv::Mat blur_img_x=img.clone();
    cv::Mat final_blur_img=img.clone();

    int img_size = img.rows;
    int kernal_radius=int((gaussian_kernel_size-1)/2);

    int start_pixel_x=kernal_radius;
    int start_pixel_y=kernal_radius;

    int end_pixel_x=  img_size - kernal_radius;
    int end_pixel_y=  img_size - kernal_radius;

    double temp_sum_res=0;
    int temp_kernal_shift=0;
    int temp_pixcel_val=0;
    double temp_gussian_val=0;

    //apply 1D filter on  x direction of the image
    for (int y=start_pixel_y;y<end_pixel_y;y++)
    {
          for(int x=start_pixel_x;x<end_pixel_x;x++)
          {
              temp_sum_res=0;
              temp_kernal_shift=0;
              for (int kernal_shift=x-kernal_radius;kernal_shift< x+kernal_radius+1;kernal_shift++)
              {
                  temp_pixcel_val=img.at<uchar>(kernal_shift,y);
                  temp_gussian_val=gussian_kernal.at<double>(temp_kernal_shift,0);

                  temp_sum_res+=temp_pixcel_val*temp_gussian_val;
                  temp_kernal_shift++;
              }
              blur_img_x.at<uchar>(x,y)=temp_sum_res;

          }
    }

    //apply 1D filter on the the result blurred_x image on Y direction
    for (int y=start_pixel_y;y<end_pixel_y;y++)
    {
        for(int x=start_pixel_x;x<end_pixel_x;x++)
        {
              temp_sum_res=0;
              temp_kernal_shift=0;
              for (int kernal_shift=y-kernal_radius;kernal_shift< y+kernal_radius+1;kernal_shift++)
              {
                  temp_pixcel_val=blur_img_x.at<uchar>(x,kernal_shift);
                  temp_gussian_val=gussian_kernal.at<double>(temp_kernal_shift,0);

                  temp_sum_res+=temp_pixcel_val*temp_gussian_val;
                  temp_kernal_shift++;
              }
              final_blur_img.at<uchar>(x,y)=temp_sum_res;
        }
    }
    return final_blur_img;

}

// Scale down the images by 2
cv::Mat img_level_scale_down(cv::Mat img)
{

    if(img.cols!=img.rows){
        cout << "Use Image with same width and height";
        return img;
    }


    if(img.cols%2 != 0){
        cout << "Use Image with size 2 power n";
        return img;
    }

    int new_size=img.rows/2;
    Mat result(new_size,new_size, CV_8UC(1), Scalar::all(0));
    for (int y=0;y<new_size;y++)
    {
        for(int x=0;x<new_size;x++)
        {
           result.at<uchar>(x,y)=img.at<uchar>(x*2,y*2);
        }
    }
    return result;

}

// Scale up the images by 2
cv::Mat img_level_scale_up(cv::Mat img)
{

    if(img.cols!=img.rows){
        cout << "Use Image with same width and height";
        return img;
    }


    if(img.cols%2 != 0){
        cout << "Use Image with size 2 power n";
        return img;
    }

    int new_size=img.rows*2;
    Mat result(new_size,new_size, CV_8UC(1), Scalar::all(0));
    for (int y=0;y<new_size;y++)
    {
        for(int x=0;x<new_size;x++)
        {
           result.at<uchar>(x,y)=img.at<uchar>(floor(x/2),floor(y/2));
        }
    }
    return result;

}

// find differences between two images
cv::Mat minus_images(cv::Mat img1,cv::Mat img2)
{

    if(img1.cols!=img2.cols){
        cout << "Tow images should be  with same width and height";
        return img1;
    }


    if(img1.cols%2 != 0){
        cout << "Use Image with size 2 power n";
        return img1;
    }

    int img_size=img1.rows;
    Mat result(img_size,img_size, CV_8SC(1), Scalar::all(0));
    for (int y=0;y<img_size;y++)
    {
        for(int x=0;x<img_size;x++)
        {
           result.at<schar>(x,y)=img1.at<uchar>(x,y) - img2.at<uchar>(x,y) ;
        }
    }

    return result;

}

// normalize the image to 0,255 range
cv::Mat img_normalize(cv::Mat img)
{
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    minMaxLoc( img, &minVal, &maxVal, &minLoc, &maxLoc );

    int img_size=img.rows;
    Mat result(img_size,img_size, CV_8UC(1), Scalar::all(0));
    for (int y=0;y<img_size;y++)
    {
        for(int x=0;x<img_size;x++)
        {
           result.at<uchar>(x,y)=((img.at<schar>(x,y) - minVal) / ( maxVal - minVal) )* 255;
        }
    }
    return result;
}

// Main function
int main(int argc, char** argv)
{

    //Read the image
    cv::Mat img = cv::imread("/home/moaljazaery/Desktop/lena512.bmp");
    // convert to gray image
    cv::Mat gray_image;
    cv::cvtColor( img, gray_image, CV_BGR2GRAY );

    //init the variable
    vector<Mat> gaussian_pyramid;
    vector<Mat> laplacian_pyramid;
    int gaussian_kernal_size=5;
    int gaussian_kernal_segma=1;

    // Blur image Demo
    Mat blur_image;
    blur_image=GaussianBlur(gray_image,gaussian_kernal_size,gaussian_kernal_segma);
    cv::imshow("Original" , gray_image );
    cv::imshow("Blur" , blur_image );


    //Build gaussian pyramid
    while(gray_image.rows > gaussian_kernal_size)
    {
        //blur the gray image
        gray_image=GaussianBlur(gray_image,gaussian_kernal_size,gaussian_kernal_segma);
        //scale down the blur gray image
        gray_image=img_level_scale_down(gray_image);
        //Add to gaussian pyramid
        gaussian_pyramid.push_back(gray_image);
    }


    //Show gaussian images
    int size_pyramid=gaussian_pyramid.size();
    for(int i=0;i<size_pyramid;i++)
    {
        string image_title="G " + (gaussian_pyramid.at(i).rows);
        cv::imshow(image_title , gaussian_pyramid.at(i) );
    }

    //Build Laplacian pyramid
    Mat res;
    for(int i=0;i<size_pyramid-1;i++)
    {
        // calculate Li = (Gi - scale_up(Gi+1))
        res=minus_images(gaussian_pyramid.at(i), img_level_scale_up(gaussian_pyramid.at(i+1)));
        laplacian_pyramid.push_back(res);
    }

    //Show Normalized Laplacian images
    Mat normailized_img;
    for(int i=0;i<size_pyramid-1;i++)
    {
        normailized_img=img_normalize(laplacian_pyramid.at(i));
        string image_title="L " + (laplacian_pyramid.at(i).rows);
        cv::imshow(image_title , normailized_img );
    }

    cv::waitKey(0);   // Wait for a keystroke in the window
    return 0;

}
