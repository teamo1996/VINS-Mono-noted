#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

// 根据状态位，进行“瘦身”，剔除掉状态位无效的点
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    // 双指针方法
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

// 给现有的特征点设置mask，目的为了特征点的均匀化
void FeatureTracker::setMask()
{
    // 鱼眼相机情况
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255)); // 构造一个mask，初始值为255
    

    // prefer to keep features that are tracked for long time
    vector<pair<int,    // 特征点被追踪的次数
    pair<cv::Point2f,   // 特征点在当前帧的坐标
    int>>> cnt_pts_id;  // 特征点的id

    // 赋值
    for (unsigned int i = 0; i < forw_pts.size(); i++){
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));
    }

    // 根据被追踪的次数对于特征点进行排序
    // 利用光流特点，追踪多的稳定性好，排前面
    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b) // lambda表达式
         {
            return a.first > b.first;
         });

    // 清空当前帧的信息
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    // 便利每一个特征点
    // 完成后同时对当前帧的特征点信息根据被跟踪次数进行了排序
    for (auto &it : cnt_pts_id)
    {
        // 如果某个像素坐标未被标记（值为255），则标记对应像素点周围一定半径圆内的值为0
        if (mask.at<uchar>(it.second.first) == 255)
        {
            // 将被遍历到的特征点重新放进容器
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            // opencv函数，把周围一个圆内全部置0,这个区域不允许别的特征点存在，避免特征点过于集中
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

// 把新的点加入容器，id给-1作为区分
void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);          // -1 代表未分配id
        track_cnt.push_back(1);
    }
}

/**
 * @brief 
 * 
 * @param[in] _img 输入图像
 * @param[in] _cur_time 图像的时间戳
 * 1、图像均衡化预处理
 * 2、光流追踪
 * 3、提取新的特征点（如果发布）
 * 4、所有特征点去畸变，计算速度
 */
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    // 进行图像均衡化
    if (EQUALIZE)
    {
        // 图像太暗或者太亮，提特征点比较难，所以均衡化一下
        // ! opencv 函数看一下
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    // 这里forw表示当前，cur表示上一帧，prev_img没有使用
    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }
    // 清空当前图像的特征点
    forw_pts.clear();

    // 如果上一帧有特征点，就可以进行光流追踪了
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        // 调用opencv函数进行光流追踪
        // Step 1 通过opencv实现多层光流追踪
        cv::calcOpticalFlowPyrLK(cur_img, forw_img,
                                 cur_pts, forw_pts,
                                 status, err,
                                 cv::Size(21, 21),
                                 3);    // 4层金字塔
        // Step 2 通过图像边界剔除outlier
        for (int i = 0; i < int(forw_pts.size()); i++)
        {
            // 判断被追踪到的特征点是否在图像范围内
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        }

        // 根据状态位，对特征点进行"瘦身"
        //reduceVector(prev_pts, status); // 没用到
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);

        reduceVector(ids, status);          // 特征点的id
        reduceVector(cur_un_pts, status);   // 去畸变后的坐标
        reduceVector(track_cnt, status);    // 追踪次数
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }
    // 被追踪到的是上一帧就存在的，因此追踪数+1
    for (auto &n : track_cnt)
        n++;

    // 如果这一帧需要被发送给后端
    if (PUB_THIS_FRAME)
    {
        // Step 3 通过对级约束来剔除outlier
        rejectWithF();

        ROS_DEBUG("set mask begins");
        TicToc t_m;

        // 为了进行新的特征点提取且确保提取到的特征点的均匀，需要设置一个mask覆盖已经被提取过特征点的区域
        setMask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        // 判断需要新提取特征点的数目
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            // 等价于mask.total()==0 代表没有任何余地可以用来提取特征点
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;

            // 只有发布才可以提取更多特征点，同时避免提的点进mask
            // 使用OpenCV的接口进行特征点提取
            cv::goodFeaturesToTrack(forw_img,
                                    n_pts,
                                    MAX_CNT - forw_pts.size(),
                                    0.01,   // 表示被提取特征点的最低质量与最佳特征质量之间的阈值比例 min = best × level
                                    MIN_DIST,          // 特征点之间的最小像素距离
                                    mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        // 将新提取的特征点加入当前帧信息
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    //prev_img = cur_img;
    //prev_pts = cur_pts;
    //prev_un_pts = cur_un_pts;   // 以上三个量无用

    cur_img = forw_img; // 实际上是上一帧的图像
    cur_pts = forw_pts; // 上一帧的特征点

    // 当前帧所有点统一去畸变，同时计算特征点速度，用来后续时间戳标定
    undistortedPoints();
    prev_time = cur_time; // 保存上一帧的时间，用于计算特征点的速度
}

/**
 * @brief 通过对极约束剔除外点
 * 
 */
void FeatureTracker::rejectWithF()
{
    // 当前被追踪到的光流至少8个点
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        // 去畸变后的上一帧特征点和去畸变后的当前帧的特征点
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        // 遍历上一帧和当前帧的特征点
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;
            // 对上一帧特征点去畸变，并投影到归一化平面
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);

            // 使用虚拟相机内参将归一化平面的点投影到虚拟像素平面
            // FOCAL_LENGTH = fx = fy = 460
            // 这里用一个虚拟相机，原因同样参考https://github.com/HKUST-Aerial-Robotics/VINS-Mono/issues/48
            // 这里有个好处就是对F_THRESHOLD和相机无关
            // 投影到虚拟相机的像素坐标系
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            // 对当前帧特征点去畸变，并投影到归一化平面
            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        // opencv接口计算本质矩阵，某种意义也是一种对级约束的outlier剔除
        // p2^TFp1 = 0
        cv::findFundamentalMat(un_cur_pts, un_forw_pts,
                               cv::FM_RANSAC,
                               F_THRESHOLD,     // 阈值为1个像素
                               0.99, status);

        // 利用本质矩阵的重投影误差，继续剔除外点
        int size_a = cur_pts.size();
        //reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

/**
 * @brief 
 * 
 * @param[in] i 
 * @return true 
 * @return false 
 *  给新的特征点赋上id,越界就返回false
 */
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;   // ！这里有个问题就是：如果系统一直运行，这个值不会越界吗？
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    // 读到的相机内参赋给m_camera
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

// 当前帧所有点统一去畸变，同时计算特征点速度，用来后续时间戳标定
void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    // TODO 这里应该先对id为-1的点的Id进行更新后再进行速度计算
    // TODO 否则，当下一帧到来时，上一帧新加的点是无法在map中找到的
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        // 对所有特征点去畸变
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        // id->坐标的map
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);

    }
    // 计算特征点速度
    // 如果上一帧不为空
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                // 根据id找到同一个特征点
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    // 得到在归一化平面的速度
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        // 第一帧的情况
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
