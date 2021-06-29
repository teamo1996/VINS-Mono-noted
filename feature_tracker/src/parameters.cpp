#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

// 读配置参数，通过roslaunch文件的参数服务器获得
void readParameters(ros::NodeHandle &n)
{
    std::string config_file;
    // 首先获得配置文件的路径
    config_file = readParam<std::string>(n, "config_file");
    // 使用opencv的yaml文件接口来读取文件
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder");

    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];            // 单帧图像的最大特征点数目
    MIN_DIST = fsSettings["min_dist"];          // 特征点之间的最短像素距离（进行均匀化）
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];                  // 发布追踪结果的频率，一个好的估计至少需要10Hz，如果设置为0,则发布频率和图像频率一致
    F_THRESHOLD = fsSettings["F_threshold"];    // 对极约束求解F矩阵时判断是否是内点的阈值
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];          // 是否做均衡化处理（主要为了避免图像太亮或者太暗，影响特征点的提取）
    FISHEYE = fsSettings["fisheye"];            // 是否为鱼眼相机
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";

    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}
