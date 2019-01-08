#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "error_code.h"
#include "common_define.h"
#include "face_detect_fb.h"
#include "keypoint.h"
#include "align.h"

using namespace std;
using namespace cv;
using namespace face_detect_fb;
using namespace face_keypoint;

#define USE_CAMERA 0
#define USE_VIDEO 0
#define USE_GPU_MAT true

static const std::string base_dir = "/home/yaoqiang/env/test";
// static const std::string image_path = "/home/wenjun/darray_det/Age_Gender/src/Picasa2/Picasa_ChenWu/Mama_F_51/img00001.jpg";
string image_path;
static const std::string video_path = base_dir + "/meinv.flv";
static const std::string fb_detect_model = base_dir + "/models/detect_fb_.bin";
static const std::string keypoint_model = base_dir + "/models/landmark_.bin";
static const int max_face_per_image = 1;
int count_name = 0;

typedef cv::Mat cpu_mat_t;
typedef cv::cuda::GpuMat gpu_mat_t;

float getMean(float* keypoints, int begin, int end)
{
    float m = 0.0, s = 0.0;
    for (int i = begin; i < end + 1; i++)
    {
        s += keypoints[i];
    }
    return s / (end - begin + 1);
}

int CreateDirectoryEx( char *sPathName )  
{  
    char DirName[256];      
    strcpy(DirName,sPathName);      
    int i,len = strlen(DirName);      
    if(DirName[len-1]!='/')      
    strcat(DirName,"/");            
    len = strlen(DirName);      
    for(i=1;i<len;i++)      
    {      
        if(DirName[i]=='/')      
        {      
            DirName[i] = 0;      
            // if(access(DirName,NULL) != 0)      
            // {      
            //     if(mkdir(DirName,0755) == -1)      
            //     {       
            //         perror("mkdir error");       
            //         return -1;       
            //     }      
            // }
            int is_exit = access(DirName, F_OK);

            if (is_exit == -1)
            {
                mkdir(DirName, 0755);
            }    
            DirName[i] = '/';      
         }      
  }        
  return 0;      
}  

template<class matT>
void face_detect_keypoint(keypoint_instance *kpt_instance, const vector<matT> &images, const vector<cv::Rect> &faces, const vector<int> &face_num, vector<float> &kpt, vector<float> &confidence)
{
    std::vector<matT> _face_mats;
    std::vector<matT> _face_original;
    vector<cv::Rect> face_r;
    for (int i = 0; i < images.size(); i++){
        for (int j = 0; j < face_num[i]; j++){
            _face_mats.push_back(images[i](faces[i * max_face_per_image + j]).clone());
            _face_original.push_back(images[i]);
            face_r.push_back(faces[i * max_face_per_image + j]);
        }
    }
    if(_face_mats.empty())
        return;
    vector<float> kpt_original;
    kpt.resize(_face_mats.size() * 68 * 2);
    kpt_original.resize(_face_original.size()*68*2);
    confidence.resize(_face_mats.size());
    vector<float> occlusion(_face_mats.size() * 68, 0.0f);
    vector<float> angle(_face_mats.size() * 3, 0.0f);
    FACE_ERROR_E error = keypoint_extract<matT>(kpt_instance, _face_mats.data(), (int)_face_mats.size(), kpt.data(), confidence.data(), occlusion.data(), angle.data(), false);
    if(error != FACE_OK){
        std::cout << "fail kpt, error = " << error << std::endl;
        assert(error == FACE_OK);
    }

    for (int j = 0; j<_face_mats.size();j++)
    {
        for(int i =136*j;i<136*(j+1);i++)
        {
            if (i<136*j +68)
            {
                kpt_original[i] = kpt[i] +face_r[j].x;
            }
            else
            {
                kpt_original[i] = kpt[i] +face_r[j].y;
            }
        }
    }

    vector<matT> align_img;
    align_img.resize(_face_original.size());
    FACE_ERROR_E error1 = face_align::align<matT>(_face_original.data(),_face_original.size(),kpt_original.data(),68,align_img.data());
    if(error != FACE_OK){
        std::cout << "fail align, error = " << error << std::endl;
        assert(error1 == FACE_OK);
    }
    string write_path ;
    write_path = image_path.replace(image_path.find("pics"),4,"pics_align");
    cout << write_path <<endl;
    string result;
    int pos = write_path.rfind("/");
    if (pos != string::npos)
    {
        result.assign(write_path,  0, + pos);
        cout << result <<endl;
    }
    CreateDirectoryEx(const_cast<char *>(result.c_str()));

    cout<<"write path:  "<<write_path<<endl;
    for (int i = 0; i < align_img.size(); i++)
    {
        cpu_mat_t cpu_align(align_img[i]);
        std::ostringstream name;
        name<<write_path;
        cv::imwrite(name.str(),cpu_align);
    }    

    // string txt_path;
    // txt_path = image_path.replace(image_path.find(".jpg"),4,".txt");
    // cout<<txt_path<<endl;
    // cout<<kpt.size()<<endl;
    // fstream out("/home/wenjun/svn/linux/demo/Detect/txt_path.txt");
    // for (int i = 0;i<135;i++)
    // // while(true)
    // {
    //     out<<kpt[i]<<"  ";
    // }
    // out.close();
}

template<class matT>
void face_detect_multi(int thread, fb_detect_instance<matT> *instance, vector<matT> &images, vector<cv::Rect> &faces, vector<float> &scores, vector<int> &face_num)
{
    clock_t t1, t2;
    FACE_ERROR_E error;

    faces.resize(max_face_per_image * images.size());
    scores.resize(max_face_per_image* images.size());
    face_num.assign(images.size(), max_face_per_image);

    t1 = clock();
    error = fb_detect_face<matT>(
            instance,
            images.data(),
            (int)images.size(),
            faces.data(),
            scores.data(),
            face_num.data(),
            false
            );
    t2 = clock();
    if(error != FACE_OK){
        std::cout << "fail detect, error = " << error << std::endl;
        assert(error == FACE_OK);
    }
    int face_detected = 0;
    for (auto &it : face_num){
        face_detected += it;
    }

    // if (face_detected==1){
    //     ofstream f; 
    //     f.open("a.txt",ios::app);
    //     f<< image_path<<endl;
    //     f.close();
    // }

    static std::mutex _lock;
    _lock.lock();
    if (typeid(matT) == typeid(cpu_mat_t)){
        std::cout << thread << " cpu mat: count = " << images.size() << " time = " << (t2 - t1)/1000 << " ms" << " face number = " << face_detected << "  fps(ms) = " << (float)(t2 - t1)/1000 / images.size() << std::endl;
    } 
    else
    {
        std::cout << thread << " gpu mat: count = " << images.size() << " time = " << (t2 - t1) / 1000 << " ms" << " face number = " << face_detected << "  fps(ms) = " << (float)(t2 - t1) / 1000 / images.size() << std::endl;
    }
    _lock.unlock();
}

void show(int thread, std::vector<cpu_mat_t> &images, const vector<cv::Rect> &_out_faces, vector<float> &scores, const vector<int> &_out_face_num, const vector<float> &kpt, const vector<float> &confidence)
{
    int face_index = 0;
    for (int i = 0; i < images.size(); i++){
        for (int j = 0; j < _out_face_num[i]; j++){
            for (int k = 0; k < 68; k++){
                cv::Point _position(cvRound(kpt[face_index * 68 * 2 + k]), cvRound(kpt[face_index * 68 * 2 + 68 + k]));
                _position.x += _out_faces[i * max_face_per_image + j].x;
                _position.y += _out_faces[i * max_face_per_image + j].y;
                cv::circle(images[i], _position, 1, cv::Scalar(0, 255, 0));
            }
            cv::rectangle(images[i], _out_faces[i * max_face_per_image + j], Scalar(0, 255, 255));
            std::ostringstream oss;
            oss << scores[i * max_face_per_image + j];
            cv::putText(images[i], oss.str(), _out_faces[i * max_face_per_image + j].br(), FONT_HERSHEY_SIMPLEX, 0.4, cvScalar(0, 0, 255));
            face_index++;
        }
        std::ostringstream oss;
        oss << "image_" << i << "_" << thread;
        cv::imshow(oss.str(), images[i]);
    }
    cv::waitKey(1000);
}

void show1(int thread, std::vector<cpu_mat_t> &images, const vector<cv::Rect> &_out_faces, vector<float> &scores, const vector<int> &_out_face_num)
{
    int face_index = 0;
    string write_path ;
    write_path = image_path.replace(image_path.find("0"),1,"1");
    // cout<<image_path<<endl;
    // cout<<write_path<<endl;
    for (int i = 0; i < images.size(); i++){
        for (int j = 0; j < _out_face_num[i]; j++){
            cv::rectangle(images[i], _out_faces[i * max_face_per_image + j], Scalar(0, 255, 255));
            std::ostringstream oss;
            oss << scores[i * max_face_per_image + j];
            cv::putText(images[i], oss.str(), _out_faces[i * max_face_per_image + j].br(), FONT_HERSHEY_SIMPLEX, 0.4, cvScalar(0, 0, 255));
            face_index++;
        }
        if (_out_face_num[i]>0){
            std::ostringstream oss;
            oss << write_path;
            cv::imwrite(oss.str(), images[i]);
        }
    }
    // cv::waitKey(0);
}

static cpu_mat_t source_img;
void get_input_images(std::vector<cpu_mat_t> &images, int count)
{
#if USE_VIDEO
    static VideoCapture _capture;
#if USE_CAMERA
    if (!_capture.isOpened())
        _capture.open(0);
#else
    if (!_capture.isOpened())
        _capture.open(video_path);
#endif
#endif

    for (int i = 0; i < count; i++){
        cpu_mat_t _img;
#if USE_VIDEO
        _capture >> _img;
        //cv::imshow("source", _img);
        //cv::waitKey(0);
#else
        _img = source_img.clone();
#endif
        images.push_back(_img);
    }
}

std::mutex _lock;

void detect_thread(int thread, int device_index, bool use_gpu_mat, int image_count)
{
    FACE_ERROR_E error = FACE_OK;

    keypoint_instance *_kpt_instance = nullptr;
    fb_detect_instance<cpu_mat_t> *_cpu_instance = nullptr;
    fb_detect_instance<gpu_mat_t> *_gpu_instance = nullptr;

    if (device_index >= 0){
        cv::cuda::setDevice(device_index);
    }

    error = keypoint_init(&_kpt_instance, device_index, keypoint_model.c_str());
    if(error != FACE_OK){
        std::cout << "fail keypoint init, error = " << error << std::endl;
        assert(error == FACE_OK);
    }

    fb_detect_param _param;
    _param.device_index = device_index;
    _param.face_confidence = 0.9f;
    error = fb_detect_init<cpu_mat_t>(&_cpu_instance, &_param, fb_detect_model.c_str());
    if(error != FACE_OK){
        std::cout << "fail fb_cpu init, error = " << error << std::endl;
        assert(error == FACE_OK);
    }

    error = fb_detect_init<gpu_mat_t>(&_gpu_instance, &_param, fb_detect_model.c_str());
    if(error != FACE_OK){
        std::cout << "fail fb_gpu init, error = " << error << std::endl;
        assert(error == FACE_OK);
    }


    ifstream txt("/home/yaoqiang/data/data_collection/live_detect/pic_fail/img_path2.txt");
    // string image_path;
    while(getline(txt,image_path))
    {
        cout <<image_path<<endl;    
        source_img = imread(image_path);

    // while (true){
        for (int i = 0; i < 1; i ++){
            std::vector<cpu_mat_t> _input_images;
            get_input_images(_input_images, image_count);

            vector<cv::Rect> _out_faces;
            vector<float> _out_scores;
            vector<int> _out_face_num;

            vector<float> kpt;
            vector<float> confidence;
            if (use_gpu_mat){
                std::vector<gpu_mat_t> gpu_input_imgs;
                for (auto &it : _input_images){
                    gpu_mat_t _gpu_img(it);
                    gpu_input_imgs.push_back(_gpu_img);
                }
                face_detect_multi<gpu_mat_t>(thread, _gpu_instance, gpu_input_imgs, _out_faces, _out_scores, _out_face_num);
                face_detect_keypoint<gpu_mat_t>(_kpt_instance, gpu_input_imgs, _out_faces, _out_face_num, kpt, confidence);
            }
            else{
                face_detect_multi<cpu_mat_t>(thread, _cpu_instance, _input_images, _out_faces, _out_scores, _out_face_num);
                face_detect_keypoint<cpu_mat_t>(_kpt_instance, _input_images, _out_faces, _out_face_num, kpt, confidence);
            }

            // show(thread, _input_images, _out_faces, _out_scores, _out_face_num, kpt, confidence);
            // show1(thread, _input_images, _out_faces, _out_scores, _out_face_num);
        }
    }
    fb_detect_destroy<cpu_mat_t>(_cpu_instance);
    fb_detect_destroy<gpu_mat_t>(_gpu_instance);
    keypoint_destroy(_kpt_instance);
}

int main(int argc, char *argv[])
{
    int device_index = 0;
    int image_count = 1;

    if (argc >= 2){
        device_index = stoi(argv[1]);
    }
    if (argc >= 3){
        image_count = stoi(argv[2]);
    }

    std::cout << "xxxx.exe " << device_index << " " << image_count << std::endl;
    
        std::vector<std::thread> _ths;
        for (int i = 0; i < 1; i ++)
            _ths.push_back(std::thread(detect_thread, i, device_index, USE_GPU_MAT, image_count));
        for (auto &it : _ths)
        {
        it.join();
        }

    //detect_thread(1, -1, USE_GPU_MAT, image_count);
    return 0;
}
