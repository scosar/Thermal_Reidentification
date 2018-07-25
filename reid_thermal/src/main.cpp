// Optris
#include <libirimager/ImageBuilder.h>
// ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>
// C++ STD
#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <cmath>
#include <limits>

#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <dirent.h>
// OpenCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "svm.h"

// Optris
evo::ImageBuilder image_builder_;
image_transport::Publisher reid_pub_;
ros::Publisher pub_phy, pub_phy_level;
unsigned char* thermal_buffer_ = NULL;
double** temperature_map_;

// ROS parameters:
double temperature_min_;
double temperature_max_;
int contour_area_min_;
int componentThr;
int componentWidth;

// Face detection
cv::Mat binary_image_;
bool personDetected = false;
// FPS
clock_t start_time_;
float start_second_;
int start_min_;
float process_second_;
int process_min_;
double fps_times_[11];
int fps_size_ = 0;

// Establish the number of bins
int histSize = 10;
// Set the ranges
float range[] = { 0, 32, 32.8, 33.6, 34.4, 35.2, 36, 36.8, 37.6, 38.4, 39.2 };
//float range[] = { 0, 30, 30.8, 31.6, 32.4, 33.2, 34, 34.8, 35.6, 36.4, 37.2, 38, 38.8 } ;
const float* histRange = { range };
bool uniform = false; bool accumulate = false;
cv::Mat temp_hist;
cv::Mat data_hist;
cv::Mat data_cdf;
cv::Mat testFeature;
cv::Mat trainTDM;
cv::Mat trainTDMInfo;
cv::Mat trainData;
cv::Mat testData;
cv::Mat testLabel;
std::vector<std::vector<int > > trainInstanceInds;
char tags[15][10] = {"Anestis", "Claudio", "Davide", "Halvard", "Jaime", "Jaycee", "Kevin", "Manuel", "Max", "Mohamed", "Nicola", "Peter", "Petra", "Sergi", "Serhan"};
double prob_estimates[15] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};

int FEATURE_SIZE = 120;
struct svm_node *svm_node_;
struct svm_model *svm_model_;
bool use_svm_model_;
bool is_probability_model_;
int predictedID = -1;
double predictedConf = 0.0;


void writeTXTasCSV(std::string filename, cv::Mat data)
{
    float valFloat;
    unsigned char valUChar;
    int type = data.type();

    std::ofstream file;
    file.open(filename.c_str());

    if (data.rows == 1)
    {

        for (int i=0;i<data.cols;i++)
        {
            valUChar = data.at<unsigned char>(i);
            valFloat = data.at<float>(i);
            if (type == CV_32FC1)
            {
                file << valFloat << ", ";
            }
            if (type == CV_8UC1)
            {
                file << static_cast<unsigned>(valUChar) << ", ";
            }
        }
    }
    else if (data.cols == 1)
    {
        for (int i=0;i<data.rows;i++)
        {
            valUChar = data.at<unsigned char>(i);
            valFloat = data.at<float>(i);
            if (type == CV_32FC1)
            {
                file << valFloat << ", ";
            }
            if (type == CV_8UC1)
            {
                file << static_cast<unsigned>(valUChar) << ", ";
            }
        }
    }
    else
    {
        for (int i=0;i<data.rows;i++)
        {
            for (int j=0; j< data.cols; j++)
            {
                valUChar = data.at<unsigned char>(i,j);
                valFloat = data.at<float>(i,j);
                if (type == CV_32FC1)
                {
                    file << valFloat << ", ";
                }
                if (type == CV_8UC1)
                {
                    file << static_cast<unsigned>(valUChar) << ", ";
                }
            }
            file << std::endl;
        }
    }


    file.close();
}

IplImage* ReadMatrixfromFile(const char* filename,char delimiter)
{
    // This function is different from the function in /aydan_cem_algo/utilities.cpp
    // This function can handle files in which elements are separated by a delimiter such as ',' or ';'. Function can also find the size of the files automatically
    std::ifstream in(filename);
    if (in.fail())
        std::cout << "File opening failed!" << std::endl;
    std::string value;
    double data;
    int w=1;
    int h=0;

    getline(in,value);
    for(int i=0;i<value.size();i++)
    {
        if (value[i]==delimiter)
            w++;

    }
    while (!in.eof())
    {
        getline(in,value);
        h++;

    }
    in.close();
    IplImage* out = cvCreateImage(cvSize(w,h),IPL_DEPTH_32F, 1);

    in.open(filename);
    std::string temp;
    int out_y=0;
    int out_x;
    while (!in.eof())
    {
        out_x=0;
        getline(in,value);
        //cout << value<< endl;
        for (int i=0;i<value.size();i++)
        {
            temp.push_back(value[i]);
            if (value[i]==delimiter)
            {

                temp.erase(temp.size()-1,1);
                //cout << temp.c_str()<< endl;
                std::istringstream a(temp);

                size_t found = a.str().find("_52");
                if (found!=std::string::npos)
                {
                    std::string id_str=a.str().substr(0,found);
                    std::string id_mod = "520" + id_str;
                    data = atof(id_mod.c_str());

                }
                else
                {
                    a >> data;
                }
                //data=atof(temp.c_str());
                //data=strtod(temp.c_str(),NULL);
                //cout << strerror(errno);
                //cout << "|"<<data<< endl;
                cvSet2D(out,out_y,out_x,cvScalar(data));
                out_x++;
                temp.erase();

            }
            if (i+1==value.size())
            {
                data=strtod(temp.c_str(),NULL);
                cvSet2D(out,out_y,out_x,cvScalar(data));
                out_x++;
                temp.erase();

            }

        }
        out_y++;

    }
    in.close();
    return out;
}

void colorConvert(const cv::Mat& raw_image, cv::Mat& color_image) {

    cv::Mat hsv_img = cv::Mat(color_image.rows,color_image.cols,color_image.type(),cv::Scalar(127,255,255));
    for (int x=0;x<raw_image.cols;x++)
    {
        for(int y=0;y<raw_image.rows;y++)
        {
            float val = raw_image.at<float>(y,x);
            //            std::cout << "val: " << val << std::endl;
            float new_val = (float) (val - 20.0) / 40.0;
            //            std::cout << "norm. val: " << new_val << std::endl;

            cv::Vec3b hsv = hsv_img.at<cv::Vec3b>(y,x);
            //            std::cout << "hsv val: " << static_cast<unsigned>(hsv.val[0]) << std::endl;
            hsv.val[0] = (uchar) (new_val * 255);
            //            std::cout << "new val: " << static_cast<unsigned>(hsv.val[0]) << std::endl;
            hsv_img.at<cv::Vec3b>(y,x) = hsv;
        }
    }

    cv::cvtColor(hsv_img,color_image,CV_HSV2RGB);
}

void colorConvert(const sensor_msgs::ImageConstPtr& raw_image, sensor_msgs::Image& color_image) {
  unsigned short* data = (unsigned short*)&raw_image->data[0];
  image_builder_.setData(raw_image->width, raw_image->height, data);

  if(thermal_buffer_ == NULL)
    thermal_buffer_ = new unsigned char[raw_image->width * raw_image->height * 3];

  image_builder_.convertTemperatureToPaletteImage(thermal_buffer_, true);

  color_image.header.frame_id = "physiological_monitoring";
  color_image.height          = raw_image->height;
  color_image.width           = raw_image->width;
  color_image.encoding        = "rgb8";
  color_image.step            = raw_image->width * 3;
  color_image.header.seq      = raw_image->header.seq;
  color_image.header.stamp    = ros::Time::now();

  color_image.data.resize(color_image.height * color_image.step);
  memcpy(&color_image.data[0], &thermal_buffer_[0], color_image.height * color_image.step * sizeof(*thermal_buffer_));
}

void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            //            std::cout << row[x] << std::endl;
            if(row[x] != 255) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);

            std::vector <cv::Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j,i));
                }
            }

            blobs.push_back(blob);

            label_count++;
        }
    }
}

void calcHistogram(cv::Mat data, cv::Mat *data_hist, cv::Mat *data_cdf, float *binVals, int nBins)
{
    data_cdf->setTo(cv::Scalar(0));

    float temp = 0;
    for(int binId=0; binId<nBins+1;binId++)
    {
        float range = binVals[binId+1] - binVals[binId];
//        std::cout << "range: " << range << std::endl;
//        std::cout << "range/2: " << range/2 << std::endl;

        data_hist->at<float>(binId,0)=0;
        data_hist->at<float>(binId,1)=binVals[binId];

//        std::cout << "Checking if data is between " << binVals[binId]-(range/2) << " and " << binVals[binId]+(range/2) << " for bin:" << binId << std::endl;
        for (int Id=0;Id<data.rows;Id++)
        {
            float val = data.at<float>(Id);

            if (val > (binVals[binId]-(range/2)) && val <= (binVals[binId]+(range/2)))
            {
                data_hist->at<float>(binId,0)++;
            }
        }

//        std::cout << data_hist->at<float>(binId,0) << std::endl;

        data_hist->at<float>(binId,0) = data_hist->at<float>(binId,0) / data.rows;
        temp = temp + data_hist->at<float>(binId,0);
        data_cdf->at<float>(binId,0) = temp;
        data_cdf->at<float>(binId,1) = binVals[binId];
    }

}

void headSegmentation(cv::Mat cv_image) {
    if(fps_size_ == 11) {
        for(int i = 0; i < 10; i++)
            fps_times_[i] = fps_times_[i+1];
        fps_size_ = 10;
    }
    fps_times_[fps_size_] = double(clock() - start_time_) / CLOCKS_PER_SEC;
    fps_size_++;

    if(binary_image_.empty())
        binary_image_.create(cv_image.rows, cv_image.cols, CV_8UC1);

    for(int i = 0; i < cv_image.rows; i++) {
        for(int j = 0; j < cv_image.cols; j++) {
            if(temperature_map_[i][j] > temperature_min_ && temperature_map_[i][j] < temperature_max_)
                binary_image_.at<uchar>(i, j) = 255;
            else
                binary_image_.at<uchar>(i, j) = 0;
        }
    }


    //// Serhan's Head Segmentation
    std::vector < std::vector<cv::Point2i > > blobs;
    FindBlobs(binary_image_, blobs);

    std::vector<int> bigBlobSmallWidthInd;
    for (int i=0;i<blobs.size();i++) {
        //        std::cout << "blobs[" << i << "].size = " << blobs[i].size() << std::endl;
        if (blobs[i].size() >= componentThr) {
            int ptx[blobs[i].size()];
            for (int j=0;j<blobs[i].size();j++) {
                ptx[j] = blobs[i][j].x;
            }

            int max_ptx = *std::max_element(ptx,ptx+blobs[i].size());
            int min_ptx = *std::min_element(ptx,ptx+blobs[i].size());
            double blobWidth = max_ptx - min_ptx;

            if (blobWidth <= componentWidth)
                bigBlobSmallWidthInd.push_back(i);

        }

    }

    int blobSize = 0;
    int biggestBlobInd = -1;
    for(int i=0; i < bigBlobSmallWidthInd.size(); i++) {
        if (blobs[bigBlobSmallWidthInd[i]].size() > blobSize) {
            blobSize = blobs[bigBlobSmallWidthInd[i]].size();
            biggestBlobInd = bigBlobSmallWidthInd[i];
        }
    }

    if (biggestBlobInd > -1) {
        cv::Mat headTemp = cv::Mat::zeros(cv::Size(1,blobSize),CV_32FC1);
        for (int i=0;i<blobSize;i++) {
            int x = blobs[biggestBlobInd][i].x;
            int y = blobs[biggestBlobInd][i].y;
            headTemp.at<float>(i) = temperature_map_[y][x];
        }

        //        std::cout << "headTemp = " << headTemp << std::endl;
//        std::cout << "Calculating histogram ..." << std::endl;
        // Compute the histograms:
//        cv::calcHist( &headTemp, 1, 0, cv::Mat(), temp_hist, 1, &histSize, &histRange, uniform, accumulate );
//        //        std::cout << temp_hist.rows << std::endl;
//        for (int i=0; i < histSize; i++)
//        {
//            temp_hist.at<float>(i) = temp_hist.at<float>(i) / blobSize;
//            //            std::cout <<  temp_hist.at<float>(i) << std::endl;
//        }

        calcHistogram(headTemp,&data_hist,&data_cdf,range,histSize);
//        std::cout << "Histogram calculated..." << std::endl;
        for (int i=0; i < histSize; i++)
        {
//            std::cout << "Checking hist bin-" << i << std::endl;
            temp_hist.at<float>(i) = data_hist.at<float>(i+1,0);
//            std::cout <<  temp_hist.at<float>(i) << std::endl;
        }
//        std::cout << "temp_hist (before transpoze) = " << temp_hist << std::endl;
//        temp_hist = temp_hist.t();
//        std::cout << "temp_hist (after transpoze) = " << temp_hist << std::endl;

        std::vector<cv::Point2i> face_points = blobs[biggestBlobInd];
        cv::Rect face_box = boundingRect(face_points);
        rectangle(cv_image, face_box, cv::Scalar(255, 255, 255));

        //        std::cout << "Rect- x: " << face_box.x << " y: " << face_box.y << " w: " << face_box.width << " h: " << face_box.height << std::endl;

        personDetected = true;
    }

    //    cv::Mat output = cv::Mat::zeros(cv_image.size(), CV_8UC3);
    //    // Red color for head
    //    for(size_t j=0; j < blobs[biggestBlobInd].size(); j++) {
    //        int x = blobs[biggestBlobInd][j].x;
    //        int y = blobs[biggestBlobInd][j].y;

    //        output.at<cv::Vec3b>(y,x)[0] = 0;
    //        output.at<cv::Vec3b>(y,x)[1] = 0;
    //        output.at<cv::Vec3b>(y,x)[2] = 255;
    //    }



    //    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    //    cv::imshow( "Display window", cv_image);                   // Show our image inside it.
    //    cv::waitKey(10);

    //// Zhi's head segmentation
    // Optional: morphological transformation (very useful for Jaime's face).
//    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20), cv::Point(1, 1)); // @todo ros_param: erosion_size_
//    cv::morphologyEx(binary_image_, binary_image_, cv::MORPH_CLOSE, element);
//    std::vector<std::vector<cv::Point> > contours;
//    findContours(binary_image_.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//    int contour_area_max = 30, contour_idx = -1; // @todo ros_param
//    for(int i = 0; i < contours.size(); i++) {
//        double a = contourArea(contours[i], false);
//        if (a > contour_area_min_) {
//            if(a > contour_area_max) {
//                contour_area_max = a;
//                contour_idx = i;
//            }
//        }
//    }
//    if(contour_idx > -1) {
//        drawContours(cv_image, contours, contour_idx, cv::Scalar(0, 255, 0));

//        std::vector<int> hull_ids;
//        convexHull(contours[contour_idx], hull_ids);
//        for(int i = 0; i < hull_ids.size(); i++)
//            circle(cv_image, contours[contour_idx][hull_ids[i]], 3, cv::Scalar(255, 0, 0), 2);
//        std::vector<cv::Vec4i> defects;
//        cv::convexityDefects(contours[contour_idx], hull_ids, defects);
//        int y_max = 0;
//        std::vector<cv::Point> face_points;
//        for(std::vector<cv::Vec4i>::iterator it = defects.begin(); it != defects.end(); it++) {
//            if((*it)[3]/256 > 20) { // @todo ros_param: max_defect_depth_
//                if(y_max < contours[contour_idx][(*it)[2]].y)
//                    y_max = contours[contour_idx][(*it)[2]].y;
//                face_points.push_back(contours[contour_idx][(*it)[2]]);
//                circle(cv_image, contours[contour_idx][(*it)[2]], 3, cv::Scalar(0, 0, 255), 2);
//            }
//        }
//        if(face_points.size() > 0) {
//            for(int i = 0; i < hull_ids.size(); i++) {
//                if(contours[contour_idx][hull_ids[i]].y < y_max)
//                    face_points.push_back(contours[contour_idx][hull_ids[i]]);
//            }
//            cv::Rect face_box = boundingRect(face_points);
//            rectangle(cv_image, face_box, cv::Scalar(255, 255, 255));

//        }

//        //cv::imshow("OpenCV debug", binary_image_);
//        //cv::waitKey(3);
//        cv::Mat headTemp = cv::Mat::zeros(cv::Size(1,face_points.size()),CV_32FC1);
//        for (int i=0;i<face_points.size();i++) {
//            int x = face_points[i].x;
//            int y = face_points[i].y;
//            headTemp.at<float>(i) = temperature_map_[y][x];
//        }

//        //        std::cout << "headTemp = " << headTemp << std::endl;
//        // Compute the histograms:
//        cv::calcHist( &headTemp, 1, 0, cv::Mat(), temp_hist, 1, &histSize, &histRange, uniform, accumulate );

//        //        std::cout << temp_hist.rows << std::endl;
//        for (int i=0; i < histSize; i++)
//        {
//            temp_hist.at<float>(i) = temp_hist.at<float>(i) / face_points.size();
//            //            std::cout <<  temp_hist.at<float>(i) << std::endl;
//        }

//        temp_hist = temp_hist.t();
//    }

}

double calculateKLD(const cv::Mat& hist, const cv::Mat& refHist) {

//    std::cout << "temp_hist (in calculateKLD) = " << hist << std::endl;
    if (hist.cols != refHist.cols) {
        std::cout << "Size of histograms does not match!! (" << hist.cols << "!=" << refHist.cols << ")" << std::endl;
        return -1;
    }
    double kld = 0;
    double refval = 0;
    for (int i=0;i<hist.cols; i++) {
        if (refHist.at<float>(i) == 0) {
            refval = std::numeric_limits<double>::epsilon();
        } else {
            refval = (double) refHist.at<float>(i);
        }
        if (hist.at<float>(i) != 0) {
            //            std::cout << "refVal= " << refval << " histVal= " << hist.at<float>(i) << std::endl;
            kld += (hist.at<float>(i)*log(hist.at<float>(i) / refval));
        }
    }
    return kld;

}

void readSVMModel(const std::string filename) {

    use_svm_model_ = false;
    if((svm_model_ = svm_load_model(filename.c_str())) == NULL) {
        std::cout << "Can not load SVM model." << std::endl;
    } else {
        std::cout << "Loading SVM model from '"<< filename << "'." << std::endl;
        is_probability_model_ = svm_check_probability_model(svm_model_)?true:false;
        svm_node_ = (struct svm_node *)malloc((FEATURE_SIZE+1)*sizeof(struct svm_node)); // 1 more size for end index (-1)
        use_svm_model_ = true;

    }
}

void classifyHist(const cv::Mat& featVector) {


    if(use_svm_model_) {

        for (int i=0;i<featVector.rows; i++) {
            //            float featVal = featVector.at<float>(i);
            //            if (featVal) {
            svm_node_[i].index = i+1;
            svm_node_[i].value = featVector.at<float>(i);
            //            }
        }
        svm_node_[FEATURE_SIZE].index = -1;
        //        svm_node_[FEATURE_SIZE].value = 0;

        //        for (int i=0;i<FEATURE_SIZE+1;i++) {
        //            std::cout << "(" << svm_node_[i].index << ", " << svm_node_[i].value << ")" << std::endl;
        //        }
        //std::cerr << "test_id = " << it->id << ", number_points = " << it->number_points << ", min_distance = " << it->min_distance << std::endl;

        // predict
        if(is_probability_model_) {            
            predictedID = svm_predict_probability(svm_model_, svm_node_, prob_estimates);
            predictedConf = prob_estimates[predictedID-1];
//            std::cout << "SVM Output: " <<  predictedID << std::endl;
//            for (int i=0;i<svm_model_->nr_class;i++) {
//                std::cout << prob_estimates[i]*100 << " ";
//            }
//            std::cout << std::endl;

        }
        //        predictedID = svm_predict(svm_model_, svm_node_);

    }
}

void createSymbolicRep(const cv::Mat& hist) {

//    std::cout << "temp_hist (in createSymbolicRep) = " << hist << std::endl;

    testFeature = cv::Mat(trainInstanceInds.size(),1,CV_32FC1);
    for (int i=0;i<trainInstanceInds.size();i++) {

        std::vector<int> indices = trainInstanceInds[i];

        double dists[indices.size()];
        for (int j=0;j<indices.size();j++) {
            cv::Mat refHist = trainTDM.row(indices[j]);

            //            std::cout << "hist = " << hist << std::endl;
            //            std::cout << "refHist = " << refHist << std::endl;

            dists[j] = calculateKLD(hist,refHist);

            //            std::cout << "dists[" << j << "] = " << dists[j] << std::endl;

            //            std::string a;
            //            std::cin >> a;

        }

        auto result = std::minmax_element(dists,dists+indices.size());
        //        std::cout << "min element at: " << (result.first - &dists[0]) << '\n';
        //        int min_dists = *std::min_element(dists,dists+indices.size());
        int min_dists_ind = (result.first - &dists[0]);
        testFeature.at<float>(i) = (float) dists[min_dists_ind];

    }

}



void thermalImageCallback(const sensor_msgs::ImageConstPtr& raw_image) {
    //if(biometrics_pub_.getNumSubscribers() == 0)
    //  return;

    /*** ros image to opencv image ***/
//    cv_bridge::CvImageConstPtr cv_ptr_raw;
//    try {
//      cv_ptr_raw = cv_bridge::toCvCopy(raw_image, sensor_msgs::image_encodings::BGR8);
//    } catch(cv_bridge::Exception& e) {
//      ROS_ERROR("cv_bridge exception: %s", e.what());
//      return;
//    }
//    cv::Mat color_image_cv;
//    colorConvert(cv_ptr_raw->image,color_image_cv);

    sensor_msgs::Image color_image;
    /*** raw (temperature) image -> RGB color image ***/
    colorConvert(raw_image, color_image);

    /*** ros image to opencv image ***/
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(color_image, sensor_msgs::image_encodings::BGR8);
    } catch(cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

//    std::cout << "In callback.. " << std::endl;
    /*** temperature decoding ***/
    unsigned short* data = (unsigned short*)&raw_image->data[0];
    for(int i = 0; i < raw_image->height; i++) {
      for(int j = 0; j < raw_image->width; j++) {
        temperature_map_[i][j] = (double(data[i*raw_image->width+j]) - 1000.0f) / 10.0f;
      }
    }

    /*** Head Segmentation ***/
    cv::Mat clrImg = cv_ptr->image;
    headSegmentation(clrImg);

    if (personDetected) {
//        std::cout << "Creating symbolic representation.. " << std::endl;
        createSymbolicRep(temp_hist);

    // writeTXTasCSV("testFeature.csv",testFeature);

    //    cv::Mat trainFeature = trainData.col(0);
    //    writeTXTasCSV("trainFeature_0.csv",trainFeature);
//        std::cout << "Classifying feature.. " << std::endl;
        classifyHist(testFeature);

        std::ostringstream ssID;
        ssID << tags[predictedID-1] << " (" << std::setprecision(4) << predictedConf*100 << "%)";
        cv::putText(clrImg, "ID:" + ssID.str(), cv::Point(cv_ptr->image.cols-200, 12), 3, 0.6, cv::Scalar(255, 255, 255));

//        std::ostringstream ssProb;
//        for (int i=0;i<svm_model_->nr_class; i++) {
//            ssProb.str("");
//            ssProb.clear();
//            ssProb << prob_estimates[i]*100;
//            cv::putText(clrImg, ssProb.str(), cv::Point(cv_ptr->image.cols-60, 50+15*(i-1)), 3, 0.6, cv::Scalar(255, 255, 255));
//        }
    }
    //        std::stringstream fileout;
    //        fileout << "/media/scosar/Windows/Users/scosar/Documents/DATASETS/LCAS_Thermal_REID/head_segmented/anestis_1_onlythermal_thermal_image/color_converted_" << frNo << ".png";
    //        cv::imwrite(fileout.str().c_str(),color_image);


    // Show FPS.
    if(fps_size_ == 11) {
        double fps = 11.0 / (fps_times_[10]-fps_times_[0]);
        std::ostringstream ss;
        ss << round(fps*10)/10.0;
        cv::putText(clrImg, ss.str()+"fps", cv::Point(cv_ptr->image.cols-80, cv_ptr->image.rows-12), 3, 0.6, cv::Scalar(255, 255, 255));
    }


//    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(),"bgr8",color_image_cv).toImageMsg();
    reid_pub_.publish(cv_ptr->toImageMsg());
}

void getTrainInstances() {

    int currentInstance = 1;
    int currentDirection = 1;
    std::vector<int> indices;
    for (int i=0;i<trainTDMInfo.rows;i++) {


        int instanceNo = trainTDMInfo.at<float>(i,0);
        int directionNo = trainTDMInfo.at<float>(i,2);

        if ((instanceNo == currentInstance) & (directionNo == currentDirection)) {
            indices.push_back(i);
            //            std::cout << "Indices for Instance-" << instanceNo << " Direction-" << directionNo << ": " << i << std::endl;
        }
        else {
            trainInstanceInds.push_back(indices);
            indices.clear();
            indices.push_back(i);
            currentInstance = instanceNo;
            currentDirection = directionNo;
            //            std::cout << "Indices for Instance-" << instanceNo << " Direction-" << directionNo << ": " << i << std::endl;
            //            std::cout << "Size of Train Indices: " << trainInstanceInds.size() << std::endl;
        }
    }
    trainInstanceInds.push_back(indices);

}

int main(int argc, char **argv) {
    ros::init(argc, argv, "reid_thermal");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    int coloring_palette;
    private_nh.param<int>("coloring_palette", coloring_palette, 6);
    image_builder_.setPalette((evo::EnumOptrisColoringPalette)coloring_palette);
    image_builder_.setPaletteScalingMethod(evo::eMinMax); // auto scaling

    // Default: Optris PI-450 output image size.
    int image_height, image_width;
    private_nh.param<int>("image_height", image_height, 288);
    private_nh.param<int>("image_width", image_width, 382);
    temperature_map_ = new double*[image_height];
    for(int i = 0; i < image_height; i++)
        temperature_map_[i] = new double[image_width];

    std::string ns = ros::this_node::getName();
    ns += "/";
    nh.param(ns+"temp_thr_min", temperature_min_, double(30.0));
    nh.param(ns+"temp_thr_max", temperature_max_, double(40.0));
    nh.param(ns+"component_thr", componentThr, int(400));
    nh.param(ns+"component_width", componentWidth, int(110));

    data_hist.create(histSize+1,2,CV_32FC1);
    data_cdf.create(histSize+1,2,CV_32FC1);
    temp_hist.create(1,histSize,CV_32FC1);

    nh.param(ns+"contour_area_min", contour_area_min_, int(30));

    std::string fileTrainTDM = ros::package::getPath("reid_thermal") +  "/config/trainTDM.csv";
    std::string fileTrainTDMInfo = ros::package::getPath("reid_thermal") + "/config/trainTDMInfo.csv";
    std::string fileTrainData = ros::package::getPath("reid_thermal") + "/config/trainData.csv";
    std::string fileSVM = ros::package::getPath("reid_thermal") + "/config/svmModel_onlyDist_KL-0.02_C-1000_e-0.01_m-4000.svm";

    trainTDM = (cv::Mat) ReadMatrixfromFile(fileTrainTDM.c_str(),',');
    trainTDMInfo = (cv::Mat) ReadMatrixfromFile(fileTrainTDMInfo.c_str(),',');
    getTrainInstances();
    readSVMModel(fileSVM.c_str());
    trainData = (cv::Mat) ReadMatrixfromFile(fileTrainData.c_str(),',');


    start_time_ = clock();

    image_transport::ImageTransport it(nh);
    image_transport::Subscriber thermal_image_sub = it.subscribe("thermal_image", 100, thermalImageCallback); // raw image

    reid_pub_ = it.advertise("/reid_thermal/reid_result", 1);

    ros::spin();

    // Release storage space.
    if(thermal_buffer_)
        delete [] thermal_buffer_;

    for(int i = 0; i < image_height; i++)
        delete [] temperature_map_[i];
    delete [] temperature_map_;

    if(use_svm_model_) {
        svm_free_and_destroy_model(&svm_model_);
        free(svm_node_);
    }



}
