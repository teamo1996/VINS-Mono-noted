/*
 * vins-mono第一次作业，去畸变方式比较
 */
#include <iostream>
#include "opencv2/opencv.hpp"
#include "eigen3/Eigen/Core"
#include "tic_toc.h"
#include <vector>

//随机产生一个点集
void generatePointSet(std::vector<cv::Point2f>& pointset,cv::Rect & rect)
{
    int rows = rect.size().height;
    int cols = rect.size().width;
    pointset.clear();
    for(int i = 20; i < cols ; i += 40){
        for(int j = 0; j < rows ; j += 40 ){
            pointset.push_back(cv::Point2f(i,j));
        }
    }
}
// 从像素平面到归一化相机平面的逆向投影
void inversePorject(std::vector<cv::Point2f> &input, std::vector<cv::Point3f> &output ,
                    cv::Mat & cameraMatrix){
    output.clear();
    for(int i = 0; i< input.size(); i++){
        float u = input[i].x;
        float v = input[i].y;

        float fx = cameraMatrix.at<float>(0,0);
        float cx = cameraMatrix.at<float>(0,2);
        float fy = cameraMatrix.at<float>(1,1);
        float cy = cameraMatrix.at<float>(1,2);

        float x = (u - cx)/fx;
        float y = (v - cy)/fy;
        output.push_back(cv::Point3f(x,y,1.0));
    }
}

// 从归一化相机平面到像素平面的正向投影
void forwardPorject(std::vector<cv::Point3f> &input, std::vector<cv::Point2f> &output ,
                    cv::Mat & cameraMatrix){
    float fx = cameraMatrix.at<float>(0,0);
    float cx = cameraMatrix.at<float>(0,2);
    float fy = cameraMatrix.at<float>(1,1);
    float cy = cameraMatrix.at<float>(1,2);
    output.clear();
    for(int i = 0; i< input.size(); i++){
        float x = input[i].x;
        float y = input[i].y;

        float u = x * fx + cx;
        float v = y * fy + cy;
        output.push_back(cv::Point2f(u,v));
    }
}

// 畸变函数
void distort(std::vector<cv::Point3f>& P, std::vector<cv::Point3f>& P_dist,cv::Mat & distcoeff)
{

    float k1 = distcoeff.at<float>(0,0);
    float k2 = distcoeff.at<float>(1,0);
    float p1 = distcoeff.at<float>(2,0);
    float p2 = distcoeff.at<float>(3,0);

    P_dist.clear();
    for(int i =0 ; i < P.size(); ++ i){
        cv::Point3f p_u = P[i];
        double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

        mx2_u = p_u.x * p_u.x;                // x^2
        my2_u = p_u.y * p_u.y ;                // y^2
        mxy_u = p_u.x * p_u.y;                // xy
        rho2_u = mx2_u + my2_u;                             // r^2
        rad_dist_u = 1 + k1 * rho2_u + k2 * rho2_u * rho2_u;    // (1 + k1*r^2 + k2*r^4)

        cv::Point3f p_d;
        p_d.x = p_u.x * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u);
        p_d.y = p_u.y * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
        p_d.z = 1.0;

        P_dist.push_back(p_d);
    }

}

// opencv去畸变
void opencvUndistort(std::vector<cv::Point2f> & dist, std::vector<cv::Point2f> & undist,
                cv::Mat & cameraMatirx, cv::Mat & distcoeff){
    cv::undistortPoints(dist,undist,cameraMatirx,distcoeff,cv::Mat(),cameraMatirx);
}

// 绘制标志点
void drawPoints(cv::Mat & img,std::vector<cv::Point2f>& points,cv::Scalar color,int size){
    for(int i =0 ;i < points.size(); ++i){
        cv::circle(img,points[i],size,color);
    }
}

// vins去畸变
void distortion(cv::Point2f p_u, cv::Point2f  & d_u, cv::Mat & distcoeff)
{
    // du = pd - pu
    float k1 = distcoeff.at<float>(0,0);
    float k2 = distcoeff.at<float>(1,0);
    float p1 = distcoeff.at<float>(2,0);
    float p2 = distcoeff.at<float>(3,0);

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u.x * p_u.x;                // x^2
    my2_u = p_u.y * p_u.y;                // y^2
    mxy_u = p_u.x * p_u.y;                // xy
    rho2_u = mx2_u + my2_u;                             // r^2
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;    // (k1*r^2 + k2*r^4)
    d_u.x =  p_u.x * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
    d_u.y =  p_u.y * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

void signleUndistort(cv::Point3f & dist, cv::Point3f & undist,cv::Mat & distcoeff)
{

    float mx_d = dist.x;        // u/fx - cx/fx = (u - cx)/fx
    float my_d = dist.y;        // v/fy - cy/fy = (v - cy)/fy

    // Recursive distortion model
    // 迭代去畸变模型 基于畸变趋向于将点往中心拉，且越远离中心畸变越大
    int n = 8;
    cv::Point2f d_u;
    // step1 根据pd计算pd-pu
    distortion(cv::Point2f (mx_d, my_d), d_u,distcoeff);
    // 计算pu
    float mx_u = mx_d - d_u.x;
    float my_u = my_d - d_u.y;

    for (int i = 1; i < n; ++i)
    {
        // step2 迭代计算pd - du = pu
        distortion(cv::Point2f (mx_d, my_d), d_u,distcoeff);
        mx_u = mx_d - d_u.x;
        my_u = my_d - d_u.y;
    }
    // Obtain a projective ray
    undist.x = mx_u;
    undist.y = my_u;
    undist.z = 1.0;
}

void vinsUndistort(std::vector<cv::Point2f> & dist, std::vector<cv::Point2f> & undist,
                   cv::Mat & cameraMatirx, cv::Mat & distcoeff){
    // 将像素坐标投影到归一化平面上
    std::vector<cv::Point3f> P_dist;
    inversePorject(dist,P_dist,cameraMatirx);
    // 进行去畸变
    std::vector<cv::Point3f> P;
    undist.clear();
    for(int i = 0; i < P_dist.size(); i++){
        cv::Point3f d = P_dist[i];
        cv::Point3f u;
        signleUndistort(d,u,distcoeff);
        P.push_back(u);
    }

    // 正向投影
    forwardPorject(P,undist,cameraMatirx);
}

int main() {

    // 随机生成点
    cv::Rect rect(100,100,720,480);

    cv::Mat img(rect.size(),CV_8UC3);
    img = cv::Scalar::all(0);

    // 原始点
    std::vector<cv::Point2f> original;
    generatePointSet(original,rect);
    // 绘制原始点
    drawPoints(img,original,cv::Scalar(0,255,0),2);

    // 逆向投影到归一化平面上
    cv::Mat cameraMatrix = ( cv::Mat_<float> ( 3,3 ) << 4.616e+02, 0.0, 3.630e+02, 0.0, 4.603e+02, 2.481e+02, 0.0, 0.0, 1.0 );
    std::vector<cv::Point3f> P;
    inversePorject(original,P, cameraMatrix);

    // 进行畸变
    std::vector<cv::Point3f> P_dist;
    // 畸变参数
    cv::Mat distCoeffs = (cv::Mat_<float>(4,1)<< -2.917e-01 , 8.228e-02 , 5.333e-05 ,-1.578e-04 );
    distort(P,P_dist,distCoeffs);

    // 正向投影
    std::vector<cv::Point2f> dist;
    forwardPorject(P_dist,dist,cameraMatrix);

    drawPoints(img,dist,cv::Scalar(255,0,0),2);

    //  opencv去畸变
    TicToc t_opencv;
    std::vector<cv::Point2f> opencv_undist;
    opencvUndistort(dist,opencv_undist,cameraMatrix,distCoeffs);
    drawPoints(img,opencv_undist,cv::Scalar(0,0,255),5);
    std::cout << "opencv undistort cost :" << t_opencv.toc() << std::endl;

    // vins去畸变
    TicToc t_vins;
    std::vector<cv::Point2f> vins_undist;
    vinsUndistort(dist,vins_undist,cameraMatrix,distCoeffs);
    drawPoints(img,vins_undist,cv::Scalar(255,255,255),7);
    std::cout << "vins undistort cost :" << t_vins.toc() << std::endl;

    cv::imshow("original",img);

    float opencv_error = 0;
    float vins_error = 0 ;

    for(int i = 0; i < original.size(); i++){
        cv::Point2f p = original[i];
        opencv_error += norm((p - opencv_undist[i]));
        vins_error += norm((p - vins_undist[i]));
    }

    opencv_error /= original.size();
    vins_error /= original.size();

    std::cout << "opencv error:" << opencv_error << std::endl;
    std::cout << "vins error:" << vins_error << std::endl;
    //cv::undistortPoints(original,dist);
    cv::waitKey(0);

    return 0;
}
