#include "netTensorRT.hpp"
#include "pointcloud_io.h"
#include "DBSCAN_kdtree.h"
#include <ikd-Tree/ikd_Tree.h>
#include <graph_match/neighborhood.hpp>
#include <graph_match/matcher.hpp>
#include <graph_match/registration.hpp>
#include "graph_match/nanoflann.hpp"
#include "graph_match/KDTreeVectorOfVectorsAdaptor.h"
#include "ros/ros.h"
#include "tic_toc.hpp"
#include"common_lib.hpp"
#include <experimental/filesystem>
#include <functional>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
// #include <pcl/kdtree/kdtree.h>
#include<pcl/search/impl/kdtree.hpp>
#include <fast_euclidean_clustering.h>
#include "FEC.h"
#include <sensor_msgs/PointCloud2.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <boost/format.hpp>
#include "DBoW3/DBoW3.h"

#include<sensor_msgs/NavSatFix.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "yaml-cpp/yaml.h"
typedef vector<pcl::PointXYZI, Eigen::aligned_allocator<pcl::PointXYZI>>  PointVector;

using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor< KeyMat, float >;

Eigen::Vector3d originllh;
Eigen::Vector3d cur_pose(0, 0, 0);
int flag = 0;


Eigen::Vector3d llh2ecef(Eigen::Vector3d data) // transform the llh to ecef
{
  Eigen::Vector3d ecef; // the ecef for output
  ecef.resize(3, 1);
  double a = 6378137.0;
  double b = 6356752.314;
  double n, Rx, Ry, Rz;
  double lon = data.x() * M_PI / 180.0; // lon to radis
  double lat = data.y() * M_PI / 180.0; // lat to radis
  double alt = data.z(); // altitude
  n = a * a / sqrt(a * a * cos(lat) * cos(lat) + b * b * sin(lat) * sin(lat));
  Rx = (n + alt) * cos(lat) * cos(lon);
  Ry = (n + alt) * cos(lat) * sin(lon);
  Rz = (b * b / (a * a) * n + alt) * sin(lat);
  ecef.x() = Rx; // return value in ecef
  ecef.y() = Ry; // return value in ecef
  ecef.z() = Rz; // return value in ecef
  return ecef;
}

bool compare(const pair<pcl::PointXYZI, float> A, const pair<pcl::PointXYZI, float> B) {
    return A.second < B.second;//升序排列
}


class sem_graph {
public:
  explicit sem_graph(ros::NodeHandle *pnh);
  std::vector<cv::Mat> descriptors;
  ~sem_graph();

private:
  int cont = 0;
  double avg_time = 0;
  std::map<int, pcl::PointCloud<pcl::PointXYZI>::Ptr> segment_cloud;
  std::map<int, std::vector<pcl::PointCloud<pcl::PointXYZI>>> cluster_label;
  pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterground;
  pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterscan;

  DBoW3::Database wordbag_loop;//词袋回环
  DBoW3::Vocabulary* voc;

  std::vector<Descriptor> histoary_descriptor;
  std::vector<std::vector<Eigen::Vector4f>> histoary_node;
  std::vector<pcl::PointCloud<pcl::PointXYZL>::Ptr> histoary_cloud;

  std::vector<Eigen::Vector3d> histoary_pose;
  std::unique_ptr<InvKeyTree> polargraph_tree_;
  KeyMat polargraph_invkeys_mat_;
  KeyMat polargraph_invkeys_to_search_;

  std::unique_ptr<InvKeyTree> polarpose_tree_;
  KeyMat polarpose_invkeys_mat_;
  KeyMat polarpose_invkeys_to_search_;

  int loop_truth = 0;
  int postive_num = 0;
  int false_num = 0;
  

  std::map<int, std::vector<Eigen::Vector4f>> cluster_centroid;
  std::map<int, int> label_map;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> loop_points;
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>> node_list;
  pcl::search::KdTree<pcl::PointXYZI>::Ptr cluster_tree;
  // pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr cluster_tree;
  void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &pc_msg);
  void GPSCallback(const geometry_msgs::PoseStamped::ConstPtr &laserOdometry);
  void hull_transform(const pcl::PointCloud<pcl::PointXYZI>::Ptr &origin_cloud, const Eigen::Vector3d &p_vision, const int &r, const int &K, const float &sorcethreshold, pcl::PointCloud<pcl::PointXYZI>::Ptr &feature_cloud);
  bool detectLoopClosure_WB(Descriptor cur_describe, std::vector<Eigen::Vector4f> cur_center, std::vector<Descriptor> histoary_describe, std::vector<std::vector<Eigen::Vector4f>> histoary_center, int bag_search_num, int *loop_index);
  bool detectLoopClosure_KD(Descriptor cur_describe, std::vector<Eigen::Vector4f> cur_center, std::vector<Descriptor> histoary_describe, std::vector<std::vector<Eigen::Vector4f>> histoary_center, int bag_search_num, int *loop_index, MatrixXi *match_id, MatrixXi *linlier_id, MatrixXf *R, MatrixXf *T);
  void visualizeLoopConstraintEdge(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> loop_frames, double time_stap);
  visualization_msgs::MarkerArray visualizeMatchID(std::vector<Eigen::Vector4f> cur_center, std::vector<Eigen::Vector4f> last_center, MatrixXi matchid, double time_stap, string id_name);
  visualization_msgs::MarkerArray visualizegraphedge(std::vector<Eigen::Vector4f> cur_center, MatrixXf graph_max, double time_stap);
  visualization_msgs::MarkerArray visualizegraphedge_last(std::vector<Eigen::Vector4f> cur_center, MatrixXf graph_max, double time_stap);
  ros::NodeHandle *pnh_;
  ros::Publisher pub_;
  ros::Publisher pub_obj_;
  ros::Publisher pub_obj_last_;
  ros::Publisher pubPath_;
  ros::Publisher pubLoopConstraintEdge_;
  ros::Publisher pubmatchedge_;
  ros::Publisher pubinleredge_;
  ros::Publisher pubgraphedge_;
  ros::Publisher pubgraphedge_last_;
  ros::Publisher pubhistoarypc_;
  nav_msgs::Path laserPath;
  ros::Subscriber sub_;
  ros::Subscriber GPS_sub;
  std::unique_ptr<rangenet::segmentation::Net> net_;
};

sem_graph::sem_graph(ros::NodeHandle *pnh) : pnh_(pnh) {
     label_map[0] = 19;
  label_map[1] = 2;
  label_map[2] = 13;
  label_map[3] = 19;
  label_map[4] = 17;
  label_map[5] = 14;
  label_map[6] = 17;
  label_map[7] = 14;
  label_map[8] = 19;
  label_map[9] = 19;
  label_map[10] = 18;
  label_map[11] = 0;
  label_map[12] = 0;
  label_map[13] = 17;
  label_map[14] = 6;
  label_map[15] = 18;
  label_map[16] = 9;
  label_map[17] = 13;
  label_map[18] = 17;
  label_map[19] = 14;
  label_map[20] = 0;
  label_map[21] = 19;
  label_map[22] = 19;
  label_map[23] = 14;
  label_map[24] = 16;
  label_map[25] = 15;
  label_map[26] = 1;
  label_map[27] = 1;
  label_map[28] = 14;

  std::experimental::filesystem::path file_path(__FILE__);
  std::string model_dir = std::string(file_path.parent_path().parent_path() / "model/");
  ROS_INFO("model_dir: %s", model_dir.c_str());
  boost::format fmt_pose("%s%s");
  // voc = new DBoW3::Vocabulary((fmt_pose % model_dir % "biglabel.dbow3").str());
  // wordbag_loop.setVocabulary(*voc, false, 0);

  for(int i = 0; i < 20; i++)
  {
    segment_cloud[i].reset(new pcl::PointCloud<pcl::PointXYZI>());
  }
  cluster_tree.reset(new pcl::search::KdTree<pcl::PointXYZI>());
  downSizeFilterscan.setLeafSize(0.4, 0.4, 0.4);
  downSizeFilterground.setLeafSize(0.5, 0.5, 0.5);


  string pointCloudTopic; // points_raw 原始点云数据
  string gpsTopic;        // imu_raw 对应park数据集，imu_correct对应outdoor数据集，都是原始imu数据，不同的坐标系表示
  ros::param::get("graph/pointCloudTopic", pointCloudTopic);
  ros::param::get("graph/gpsTopic", gpsTopic);


  GPS_sub = pnh_->subscribe<geometry_msgs::PoseStamped>(gpsTopic, 100, &sem_graph::GPSCallback, this); // /kitti/oxts/gps/fix  /gps/fix

  sub_ = pnh_->subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 10, &sem_graph::pointcloudCallback, this);
  
  pub_ = pnh_->advertise<sensor_msgs::PointCloud2>("/label_pointcloud", 1, true);
  pub_obj_ = pnh_->advertise<sensor_msgs::PointCloud2>("/graph_node", 1, true);
  pub_obj_last_ = pnh_->advertise<sensor_msgs::PointCloud2>("/graph_node_last", 1, true);
  pubPath_ = pnh_->advertise<nav_msgs::Path>("/robot_path", 10);
  pubLoopConstraintEdge_ = pnh_->advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 10);
  pubmatchedge_ = pnh_->advertise<visualization_msgs::MarkerArray>("/loop_closure_match", 10);
  pubinleredge_ = pnh_->advertise<visualization_msgs::MarkerArray>("/loop_closure_inler", 10);
  pubgraphedge_ = pnh_->advertise<visualization_msgs::MarkerArray>("/graph_edge", 10);
  pubgraphedge_last_ = pnh_->advertise<visualization_msgs::MarkerArray>("/graph_edge_last", 10);
  pubhistoarypc_ = pnh_->advertise<sensor_msgs::PointCloud2>("/label_pointcloud_last", 1, true);
  // ROS_WARN("1111111111111111111111");
  net_ = std::unique_ptr<rangenet::segmentation::Net>(new rangenet::segmentation::NetTensorRT(model_dir, false));
  // ROS_WARN("22222222222222222222222");
};

sem_graph::~sem_graph(){

}

void sem_graph::pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &pc_msg) {

  // ROS_WARN("33333333333333333333333333333");
  // ROS 消息类型 -> PCL 点云类型
  pcl::PointCloud<PointLabel>::Ptr pc_ros(new pcl::PointCloud<PointLabel>());
  pcl::fromROSMsg(*pc_msg, *pc_ros);
  double cur_time = pc_msg->header.stamp.toSec();
//   for(int i=0; i < pc_ros->size(); i++)
//   {
//     pc_ros->points[i].intensity = pc_ros->points[i].intensity/65535;
//   }
  
  // 预测
//   auto labels = std::make_unique<int[]>(pc_ros->size());
//   net_->doInfer(*pc_ros, labels.get());
  pcl::PointCloud<pcl::PointXYZRGB> color_pc;
  pcl::PointCloud<pcl::PointXYZRGB> color_node;

  

  
  // 发布点云
  int point_num = pc_ros->size();
  std::vector<int> labels;
  pcl::PointXYZI temp_point;
  pcl::PointXYZL temp_point2;
  pcl::PointCloud<pcl::PointXYZI>::Ptr ground_points(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZL>::Ptr label_clouds(new pcl::PointCloud<pcl::PointXYZL>());
  std::pair<std::vector<Eigen::Vector3d>,std::vector<int>> label_points;
  pcl::PointCloud<pcl::PointXYZL>::Ptr background_points1(new pcl::PointCloud<pcl::PointXYZL>());
  std::pair<std::vector<Eigen::Vector3d>,std::vector<int>> clusterPointL_a;
  std::vector<Eigen::Vector3d> pc_out;
  std::vector<int> label_out;
  for (size_t i = 0; i < point_num; ++i) 
  {
    temp_point.x = pc_ros->points[i].x;
    temp_point.y = pc_ros->points[i].y;
    temp_point.z = pc_ros->points[i].z;
    temp_point.intensity = pc_ros->points[i].intensity;
    int label = pc_ros->points[i].label + 1;
    if(label > 19)
    {
      continue;
    }
    segment_cloud[label]->push_back(temp_point);

    ground_points->push_back(temp_point);


    uint32_t r = std::get<2>(net_->_argmax_to_rgb[label]);
    uint32_t g = std::get<1>(net_->_argmax_to_rgb[label]);
    uint32_t b = std::get<0>(net_->_argmax_to_rgb[label]);
    uint32_t rgb = (r << 16) | (g << 8) | b;
    pcl::PointXYZRGB temp_point_c;
    temp_point_c.x = temp_point.x;
    temp_point_c.y = temp_point.y;
    temp_point_c.z = temp_point.z;
    temp_point_c.rgb = *reinterpret_cast<float *>(&rgb); 
    color_pc.push_back(temp_point_c);



    temp_point2.x = temp_point.x;
    temp_point2.y = temp_point.y;
    temp_point2.z = temp_point.z;
    temp_point2.label = net_->_lable_map[label];
    label_clouds->push_back(temp_point2);

    Eigen::Vector3d temp_point3;
    temp_point3.x() = temp_point.x;
    temp_point3.y() = temp_point.y;
    temp_point3.z() = temp_point.z;

    pc_out.push_back(temp_point3);
    label_out.push_back(label);
    // if(label == 9 || label == 10 || label == 11 || label == 12 ||label == 13 || label == 17)
    // {
    //   temp_point.intensity = label;segment_cloud[17]
  }

  
  TicToc t1;
  // ROS_WARN("1111111111111111111111");
  // FastEuclideanClustering<pcl::PointXYZI> fec;
  // fec.setClusterTolerance(1);
  // fec.setMinClusterSize(10);
  // fec.setMaxClusterSize(5000);
  //合并地面
  // downSizeFilterground.setInputCloud(ground_points);
  // downSizeFilterground.filter(*ground_points);

  downSizeFilterscan.setInputCloud(segment_cloud[13]);
  downSizeFilterscan.filter(*segment_cloud[13]);

  downSizeFilterscan.setInputCloud(segment_cloud[14]);
  downSizeFilterscan.filter(*segment_cloud[14]);

  downSizeFilterscan.setInputCloud(segment_cloud[15]);
  downSizeFilterscan.filter(*segment_cloud[15]);

  downSizeFilterscan.setInputCloud(segment_cloud[17]);
  downSizeFilterscan.filter(*segment_cloud[17]);

  //聚类
  float max_cos = 0.02;
  int search_num = 100;
  float max_distance = 0.04;
  int dinzhi = 1000;
  int search_k = 20;
  float thelod = 0.7;
  std::vector<Eigen::Vector4f> cur_node_vector;
  Eigen::Vector3d p_vv(0,0,500);
  Eigen::Vector4f centroid;
  //节点提取
  for(int i = 0; i < 20; i++)
  {
    if(!segment_cloud[i]->points.empty() && ( i == 14 || i == 15 || i == 16 || i == 17 || i == 18 || i == 19))
    {

      std::vector<pcl::PointIndices> cluster_indices;
      int cluster_size = 20;
      double cluster_r = 0.5;
      int cluster_maxn = 35;
      if(i == 15 || i == 17)
      {
        cluster_size = 150;
        cluster_r = 1;
        cluster_maxn = 100;
      }
      else
      {
        cluster_size = 20;
        cluster_r = 1;
        cluster_maxn = 100;       
      }
      cluster_indices = FEC(segment_cloud[i], cluster_size, cluster_r, cluster_maxn);
      if((int)cluster_indices.size() == 0)
      {
        continue;
      }
      for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end();it++) 
      {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto &idx : it->indices) cloud_cluster->push_back((*segment_cloud[i])[idx]);

        pcl::compute3DCentroid(*cloud_cluster, centroid);
        int close_flag = 0;
        for(int j = 0; j < cluster_centroid[i].size(); j++)
        {
          Eigen::Vector4f histoary_centroid;
          histoary_centroid = cluster_centroid[i][j];
          if(sqrt((histoary_centroid[0]-centroid[0]) * (histoary_centroid[0]-centroid[0]) + (histoary_centroid[1]-centroid[1]) * (histoary_centroid[1]-centroid[1]) + (histoary_centroid[2]-centroid[2]) * (histoary_centroid[2]-centroid[2])) < 1)
          {
            close_flag = 1;
            break;
          }
        }
        if(close_flag == 1)
        {
          continue;
        }        cluster_size = 10;
        cluster_r = 0.1;
        cluster_maxn = 100;
        centroid[3] = i - 13;
        cluster_centroid[i].push_back(centroid);
        cur_node_vector.push_back(centroid);
        uint32_t r = std::get<2>(net_->_argmax_to_rgb[i]);
        uint32_t g = std::get<1>(net_->_argmax_to_rgb[i]);
        uint32_t b = std::get<0>(net_->_argmax_to_rgb[i]);
        uint32_t rgb = (r << 16) | (g << 8) | b;
        pcl::PointXYZRGB node_vision;
        node_vision.x = centroid[0];
        node_vision.y = centroid[1];
        node_vision.z = centroid[2];
        node_vision.rgb = *reinterpret_cast<float *>(&rgb);
        color_node.push_back(node_vision);
      }
      // cluster_tree->setInputCloud(segment_cloud[i]);
      // fec.setSearchMethod(cluster_tree);
      // fec.setInputCloud(segment_cloud[i]);
      // fec.setQuality(0.5);
      // fec.segment(cluster_indices);
      // printf("classes:%d, num:%ld\n", i, cluster_centroid[i].size());
    }
    if(!segment_cloud[i]->points.empty() && i == 13)
    {
      std::vector<pcl::PointIndices> cluster_indices;
      int cluster_size = 200;
      double cluster_r = 1;
      int cluster_maxn = 100;
      // pcl::PointCloud<pcl::PointXYZI>::Ptr surf_points(new pcl::PointCloud<pcl::PointXYZI>());
      // hull_transform(segment_cloud[13], p_vv, dinzhi, search_k, thelod, surf_points); 
      cluster_indices = FEC(segment_cloud[13], cluster_size, cluster_r, cluster_maxn);
      if((int)cluster_indices.size() == 0)
      {
        continue;
      } 
      for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end();it++) 
      {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZI>);
        for (const auto &idx : it->indices) cloud_cluster->push_back((*segment_cloud[13])[idx]);

        pcl::compute3DCentroid(*cloud_cluster, centroid);
        int close_flag = 0;
        for(int j = 0; j < cluster_centroid[i].size(); j++)
        {
          Eigen::Vector4f histoary_centroid;
          histoary_centroid = cluster_centroid[i][j];
          if(sqrt((histoary_centroid[0]-centroid[0]) * (histoary_centroid[0]-centroid[0]) + (histoary_centroid[1]-centroid[1]) * (histoary_centroid[1]-centroid[1]) + (histoary_centroid[2]-centroid[2]) * (histoary_centroid[2]-centroid[2])) < 1)
          {
            close_flag = 1;
            break;
          }
        }
        if(close_flag == 1)
        {
          continue;
        }
        cluster_label[i].push_back(*cloud_cluster);
        centroid[3] = i - 13;
        cluster_centroid[i].push_back(centroid);
        cur_node_vector.push_back(centroid);
        uint32_t r = std::get<2>(net_->_argmax_to_rgb[i]);
        uint32_t g = std::get<1>(net_->_argmax_to_rgb[i]);
        uint32_t b = std::get<0>(net_->_argmax_to_rgb[i]);
        uint32_t rgb = (r << 16) | (g << 8) | b;
        pcl::PointXYZRGB node_vision;
        node_vision.x = centroid[0];
        node_vision.y = centroid[1];
        node_vision.z = centroid[2];
        node_vision.rgb = *reinterpret_cast<float *>(&rgb);
        color_node.push_back(node_vision);
      }
      // printf("classes:%d, num:%ld\n", i, cluster_centroid[i].size());
    }
  }

  Neighborhood Nei1(cur_node_vector);
  MatrixXf descriptor1;
  Descriptor Des1(Nei1, 3, 1000, 1);
  cv::Mat des_mat = Des1.getDescriptor();
  
  sensor_msgs::PointCloud2 ros_msg2;
  pcl::toROSMsg(color_node, ros_msg2);
  ros_msg2.header = pc_msg->header;
  ros_msg2.header.frame_id = "base_link";
  pub_obj_.publish(ros_msg2);

  MatrixXf neigber_max = Des1.vis_edge;
  visualization_msgs::MarkerArray graph_edge = visualizegraphedge(cur_node_vector, neigber_max, cur_time);
  pubgraphedge_.publish(graph_edge);


  // // cv::FileStorage fs_out((fmt_pose % "/home/wsc/range_net/src/RangeNetTrt8/wordbag/mat" % cont % "yaml").str(),cv::FileStorage::WRITE); 
  // cv::FileStorage fs_out((fmt_pose % "/home/wsc/range_net/src/RangeNetTrt8/wordbag/mat_int" % cont % "yaml").str(),cv::FileStorage::WRITE); 
  std::vector<float> cur_pose_ve;
  cur_pose_ve.resize(3);
  cur_pose_ve[0] = cur_pose.x(); cur_pose_ve[1] = cur_pose.y(); cur_pose_ve[2] = cur_pose.z();
  if(cont > 150)
  {
    std::vector<size_t> ret( 3 ); 
    std::vector<float> out_dists_sqr( 3 );
    nanoflann::KNNResultSet<float> knnsearch_result( 3 );
    knnsearch_result.init( &ret[0], &out_dists_sqr[0] );
    polarpose_tree_->index->findNeighbors( knnsearch_result, &cur_pose_ve[0] /* query */, nanoflann::SearchParams(3) );
    for (unsigned int i = 0; i < ret.size(); i++)
    {
      if((cur_pose - histoary_pose[ret[i]]).norm() <10)
      {
        loop_truth = loop_truth+1;
        break;
      }
    }
    
    int loop_id = -1;
    MatrixXf R, T;
    R.setIdentity();
    T.setZero();
    MatrixXi matchID;
    MatrixXi inlierID;

    if(detectLoopClosure_KD(Des1, cur_node_vector, histoary_descriptor ,histoary_node ,15 ,&loop_id, &matchID, &inlierID, &R, &T))
    {

        std::pair<Eigen::Vector3d, Eigen::Vector3d> now_loop_point;
        now_loop_point.first = cur_pose;
        now_loop_point.second = histoary_pose[loop_id];
        loop_points.push_back(now_loop_point);
        visualizeLoopConstraintEdge(loop_points, cur_time);
        std::vector<Eigen::Vector4f> close_node;
        close_node = histoary_node[loop_id];
        visualization_msgs::MarkerArray graph_edge_last = visualizegraphedge_last(close_node, histoary_descriptor[loop_id].vis_edge, cur_time);
        pubgraphedge_last_.publish(graph_edge_last);
        


        sensor_msgs::PointCloud2 ros_msg3;
        pcl::toROSMsg(node_list[loop_id], ros_msg3);
        ros_msg3.header = pc_msg->header;
        ros_msg3.header.frame_id = "base_link";
        pub_obj_last_.publish(ros_msg3);

        // sensor_msgs::PointCloud2 ros_msg4;
        // pcl::toROSMsg(histoary_cloud[loop_id], ros_msg4);
        // ros_msg4.header = pc_msg->header;
        // ros_msg4.header.frame_id = "base_link";
        // pubhistoarypc_.publish(ros_msg4);

        visualization_msgs::MarkerArray match_vis = visualizeMatchID(cur_node_vector, close_node, matchID, cur_time, "match_result");
        visualization_msgs::MarkerArray inlier_vis = visualizeMatchID(cur_node_vector, close_node, inlierID, cur_time, "inlier_result");
        pubmatchedge_.publish(match_vis);
        pubinleredge_.publish(inlier_vis);
        //save pcd
        Matrix4f TransforMatrix = Matrix4f::Identity();
        for(int row = 0; row<3; row++){
            for(int col=0; col<3; col++){
                TransforMatrix(row, col) = R(row, col);
            }
        }
        TransforMatrix(0, 3) = T(0, 0);
        TransforMatrix(1, 3) = T(1, 0);
        TransforMatrix(2, 3) = T(2, 0);
        ROS_WARN("close loop %d!!!!!!!!!!!!!!!!!!!!!!!!!", loop_id);
        if((cur_pose - histoary_pose[loop_id]).norm() <=20)
        { 
          postive_num = postive_num + 1;
        }
        if((cur_pose - histoary_pose[loop_id]).norm() >=30)
        { 
          false_num = false_num + 1;
        }
    }
    else
    {
        std::vector<Eigen::Vector4f> cur_node_vector_temp;
        std::vector<Eigen::Vector4f> close_node;
        MatrixXi matchID1;
        MatrixXi inlierID1;

        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = "base_link";
        markerEdge.header.stamp = ros::Time().fromSec(cur_time);
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "inlier_result";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.6; markerEdge.scale.y = 0.6; markerEdge.scale.z = 0.6;
        markerEdge.color.r = 0.447; markerEdge.color.g = 0.623; markerEdge.color.b = 0.812;
        markerEdge.color.a = 0.8;
        visualization_msgs::MarkerArray inlier_vis;
        inlier_vis.markers.push_back(markerEdge);
        visualization_msgs::MarkerArray graph_edge_last;
        pubgraphedge_last_.publish(graph_edge_last);
        pubinleredge_.publish(inlier_vis);
        sensor_msgs::PointCloud2 ros_msg3;
        ros_msg3.header = pc_msg->header;
        ros_msg3.header.frame_id = "base_link";
        pub_obj_last_.publish(ros_msg3);
    }
    std::cout<<"all loop num :"<< loop_truth <<"postive num:"<<postive_num<<"false num:"<<false_num<<std::endl;

  }
  

  //导出描述子
  // boost::format fmt_pose("%s/%d.%s");
  // // cv::FileStorage fs_out((fmt_pose % "/home/wsc/range_net/src/RangeNetTrt8/wordbag/mat" % cont % "yaml").str(),cv::FileStorage::WRITE); 
  // cv::FileStorage fs_out((fmt_pose % "/home/wsc/range_net/src/RangeNetTrt8/wordbag/mat_int" % cont % "yaml").str(),cv::FileStorage::WRITE); 
  // fs_out << "Descriptor" << des_mat;
  // fs_out.release();


  //   descriptors.push_back(des_mat);

  //   DBoW3::Vocabulary vocab;
  //   vocab.create(descriptors);
  //   cout<<"vocabulary info: "<<vocab<<endl;
  //   vocab.save("/home/wsc/range_net/src/RangeNetTrt8/wordbag/segment_graph.dbow3");



  
  
 
  // cout<<"done"<<endl;


  int frames_size = histoary_descriptor.size();

  // wordbag_loop.add(des_mat);
  histoary_descriptor.push_back(Des1);
  histoary_node.push_back(cur_node_vector);
  histoary_pose.push_back(cur_pose);
  polargraph_invkeys_mat_.push_back(Des1.globalDescriptor);


 
  for(int i = 0; i < color_node.size(); i++)
  {
    color_node.points[i].z = color_node.points[i].z + 30;
  }
  node_list.push_back(color_node);


  polarpose_invkeys_mat_.push_back(cur_pose_ve);
  
  if( cont % 20 == 0 && cont > 100) // to save computation cost
  {
      // TicToc t_tree_construction;
      polargraph_invkeys_to_search_.clear();
      polargraph_invkeys_to_search_.assign(polargraph_invkeys_mat_.begin(), polargraph_invkeys_mat_.end() - 50) ;
      // 构建基于环键的搜索树（用于进行最近邻搜索）
      polargraph_tree_.reset(); 
      polargraph_tree_ = std::make_unique<InvKeyTree>(Des1.globalDescriptor.size() /* dim */, polargraph_invkeys_to_search_, 10 /* max leaf */ );
      
      
      polarpose_invkeys_to_search_.clear();
      polarpose_invkeys_to_search_.assign(polarpose_invkeys_mat_.begin(), polarpose_invkeys_mat_.end() - 100) ;
      // 构建基于环键的搜索树（用于进行最近邻搜索）
      polarpose_tree_.reset(); 
      polarpose_tree_ = std::make_unique<InvKeyTree>(cur_pose_ve.size() /* dim */, polarpose_invkeys_to_search_, 10 /* max leaf */ );
      // tree_ptr_->index->buildIndex(); // inernally called in the constructor of InvKeyTree (for detail, refer the nanoflann and KDtreeVectorOfVectorsAdaptor)
      // t_tree_construction.toc();
      // ROS_INFO("Tree construction");
  }
  // Neighborhood Nei1(centerpoint1);
  std::cout << "word bag add:"<< cont <<std::endl;
  cont = cont + 1;
  avg_time = avg_time + t1.toc();
  printf("predict spend time: %f ms \n", avg_time/cont);

  for(int i = 0; i < 20; i++)
  {
    segment_cloud[i]->clear();
    cluster_label[i].clear();
    cluster_centroid[i].clear();
    cur_node_vector.clear();
  }

  sensor_msgs::PointCloud2 ros_msg;
//   dynamic_cast<rangenet::segmentation::NetTensorRT *>(net_.get())->paintPointCloud(*pc_ros, color_pc, labels);
  
  pcl::toROSMsg(color_pc, ros_msg);
  ros_msg.header = pc_msg->header;
  ros_msg.header.frame_id = "base_link";
  pub_.publish(ros_msg);
//   histoary_cloud_save.push_back(color_pc);
  // for(int i = 0; i < color_pc.points.size(); i++)
  // {
  //   color_pc.points[i].z = color_pc.points[i].z + 50;
  // }
  // histoary_cloud.push_back(color_pc);

}

void sem_graph::GPSCallback(const geometry_msgs::PoseStamped::ConstPtr &laserOdometry){

  cur_pose.x() = laserOdometry->pose.position.x;
  cur_pose.y() = laserOdometry->pose.position.y;
  cur_pose.z() = laserOdometry->pose.position.z;

  geometry_msgs::PoseStamped RobotPose;
  RobotPose.header.stamp = laserOdometry->header.stamp;
  RobotPose.header.frame_id = "world";
  RobotPose.pose.orientation.x = 0;
  RobotPose.pose.orientation.y = 0;
  RobotPose.pose.orientation.z = 0;
  RobotPose.pose.orientation.w = 1;
  RobotPose.pose.position.x = cur_pose.x();
  RobotPose.pose.position.y = cur_pose.y();
  RobotPose.pose.position.z = cur_pose.z();
  laserPath.header.stamp = laserOdometry->header.stamp;
  laserPath.header.frame_id = "world";
  laserPath.poses.push_back(RobotPose);
  pubPath_.publish(laserPath);
}

void sem_graph::hull_transform(const pcl::PointCloud<pcl::PointXYZI>::Ptr &origin_cloud, const Eigen::Vector3d &p_vision, const int &r, const int &K, const float &sorcethreshold, pcl::PointCloud<pcl::PointXYZI>::Ptr &feature_cloud)
{
  float max_distance = 0;
  pcl::PointCloud<pcl::PointXYZI>::Ptr hull_cloud(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr first_cloud(new pcl::PointCloud<pcl::PointXYZI>());
  hull_cloud->points.resize(origin_cloud->size());
  for(int i = 0; i < origin_cloud->size(); i++)
  {
    float temp_distance = sqrt((origin_cloud->points[i].x - p_vision.x()) * (origin_cloud->points[i].x - p_vision.x())
                              +(origin_cloud->points[i].y - p_vision.y()) * (origin_cloud->points[i].y - p_vision.y())
                              +(origin_cloud->points[i].z - p_vision.z()) * (origin_cloud->points[i].z - p_vision.z()));
    if(temp_distance > max_distance)
    {
      max_distance = temp_distance;
    }
  }
  #pragma omp parallel for num_threads(14)
  for(int i = 0; i < origin_cloud->size(); i++)
  {
    Eigen::Vector3d p_origin(origin_cloud->points[i].x, origin_cloud->points[i].y, origin_cloud->points[i].z);
    float f = r * (max_distance - (p_origin - p_vision).norm());
    Eigen::Vector3d p_hull;
    if(p_origin != p_vision)
    {
      p_hull = f * ((p_origin - p_vision)/(p_origin - p_vision).norm());
    }
    else
    {
      p_hull = p_vision;
    }
    hull_cloud->points[i].x = p_hull.x();
    hull_cloud->points[i].y = p_hull.y();
    hull_cloud->points[i].z = p_hull.z();
    hull_cloud->points[i].intensity = origin_cloud->points[i].intensity;
  }
  pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
  kdtree.setInputCloud(hull_cloud);
  #pragma omp parallel for num_threads(14)
  for(int i = 0; i < hull_cloud->size(); i++)
  {
    // 定义临近点的索引序列，距离序列
    std::vector<int> pointIdxNKNSearch(K);		  //index in the order of the distance    k个
    std::vector<float> pointNKNSquaredDistance(K); //distance square  
    kdtree.nearestKSearch(hull_cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance); // K NN search result
    float k_means = 0;
    for(int j = 0; j < pointIdxNKNSearch.size(); j++)
    {
      k_means = k_means + sqrt((hull_cloud->points[pointIdxNKNSearch[j]].x - p_vision.x()) * (hull_cloud->points[pointIdxNKNSearch[j]].x - p_vision.x())
                              +(hull_cloud->points[pointIdxNKNSearch[j]].y - p_vision.y()) * (hull_cloud->points[pointIdxNKNSearch[j]].y - p_vision.y())
                              +(hull_cloud->points[pointIdxNKNSearch[j]].z - p_vision.z()) * (hull_cloud->points[pointIdxNKNSearch[j]].z - p_vision.z()));
    }
    k_means = k_means/K;
    float dis_h = sqrt((hull_cloud->points[i].x - p_vision.x()) * (hull_cloud->points[i].x - p_vision.x())
                      +(hull_cloud->points[i].y - p_vision.y()) * (hull_cloud->points[i].y - p_vision.y())
                      +(hull_cloud->points[i].z - p_vision.z()) * (hull_cloud->points[i].z - p_vision.z()));
    if(dis_h > k_means)
    {
      #pragma omp critical
      first_cloud->push_back(origin_cloud->points[i]);
    }
  }
  float d_mean = 0;
  float d_var = 0;
  for(int i = 0; i < first_cloud->size(); i++)
  {
    float dis_t = sqrt((first_cloud->points[i].x - p_vision.x()) * (first_cloud->points[i].x - p_vision.x())
                      +(first_cloud->points[i].y - p_vision.y()) * (first_cloud->points[i].y - p_vision.y())
                      +(first_cloud->points[i].z - p_vision.z()) * (first_cloud->points[i].z - p_vision.z()));
    d_mean += (dis_t - d_mean)/(i+1);
    d_var = d_var*i/(i + 1) + ((dis_t - d_mean) * (dis_t -d_mean)) * i/((i + 1)*(i + 1));
  }
  #pragma omp parallel for num_threads(14)
  for(int i = 0; i < first_cloud->size(); i++)
  {
    float dis_t = sqrt((first_cloud->points[i].x - p_vision.x()) * (first_cloud->points[i].x - p_vision.x())
                      +(first_cloud->points[i].y - p_vision.y()) * (first_cloud->points[i].y - p_vision.y())
                      +(first_cloud->points[i].z - p_vision.z()) * (first_cloud->points[i].z - p_vision.z()));
    float sorce = exp(-(dis_t - d_mean)*(dis_t - d_mean)/(2*d_var));
    if(sorce > sorcethreshold)
    {
      #pragma omp critical
      feature_cloud->push_back(first_cloud->points[i]);
    }
  }

}

bool sem_graph::detectLoopClosure_WB(Descriptor cur_describe, std::vector<Eigen::Vector4f> cur_center, std::vector<Descriptor> histoary_describe, std::vector<std::vector<Eigen::Vector4f>> histoary_center, int bag_search_num, int *loop_index)
{
  cv::Mat des_mat = cur_describe.getDescriptor();
  DBoW3::QueryResults ret;
  std::vector<DBoW3::QueryResults> ret_queue;
  std::vector<int> index_sort;
  wordbag_loop.query(des_mat, ret, bag_search_num, cont-150);
  if(ret.size()<1)
  {
    return false;
  }  
  bool find_loop = false;
  if (ret.size() >= 1 && ret[0].Score > 0.1)
  {
    for (unsigned int i = 0; i < ret.size(); i++)
    {
      if (ret[i].Score > 0.1)
      {          
        find_loop = true;
      }
      index_sort.push_back(ret[i].Id);
    }
  }
  else
  {
    return false;
  }
  int min_index = -1;
  if(find_loop)
  {
    for (int i = 0; i < index_sort.size(); i++)
    {
      for(int j = 0; j < index_sort.size() - i; j++)
      {
        int temp_sort;
        if(index_sort[j] > index_sort[j + 1])
        {
          temp_sort = index_sort[j + 1];
          index_sort[j + 1] = index_sort[j];
          index_sort[j] = temp_sort;
        }
      }
    }
    min_index = index_sort[0];
  }
  else
  {
    return false;
  }
  if(min_index == -1)
  {
    return false;
  }
  int index_num = 0;
  double max_score = -1;
  int max_index = 0;
  while (index_num < index_sort.size())
  {
    Descriptor last_describe = histoary_describe[index_sort[index_num]];
    MatrixXi matcherID;
    matcher matches(cur_describe, last_describe, 2);
    matcherID = matches.getGoodMatcher();
    double socre1 = (double)(matcherID.rows())/(double)(cur_describe.labelVector.size());
    if(socre1 < 0.5) // kitti 0.4
    {
      index_num =index_num + 1;
      continue;
    }
    registration registration(cur_center, histoary_center[index_sort[index_num]], matcherID);
    registration.matcherRANSAC(2, 7, 100);
    MatrixXi inlierID;
    inlierID = registration.inlierID;
    double socre = (double)(inlierID.rows())/(double)(matcherID.rows());
    if(socre >= max_score)
    {
      max_score = socre;
      max_index = index_sort[index_num];
    }
    index_num =index_num + 1;
  }
  if(max_score > 0.22) // kitti00 0.22
  {
    std::cout<< "matcher_ID" << max_index <<"matcher_score:"<< max_score <<std::endl;
    *loop_index = max_index;
    return true;
  }
  else
  {
  return false;
}
}

bool sem_graph::detectLoopClosure_KD(Descriptor cur_describe, std::vector<Eigen::Vector4f> cur_center, std::vector<Descriptor> histoary_describe, std::vector<std::vector<Eigen::Vector4f>> histoary_center, int bag_search_num, int *loop_index, MatrixXi *match_id, MatrixXi *linlier_id, MatrixXf *R, MatrixXf *T)
{
  cv::Mat des_mat = cur_describe.getDescriptor();
  // DBoW3::QueryResults ret;
  // std::vector<DBoW3::QueryResults> ret_queue;
  std::vector<size_t> index_sort;
  // wordbag_loop.query(des_mat, ret, bag_search_num, cont-150);
  std::vector<size_t> ret( bag_search_num ); 
  std::vector<float> out_dists_sqr( bag_search_num );
  nanoflann::KNNResultSet<float> knnsearch_result( bag_search_num );
  knnsearch_result.init( &ret[0], &out_dists_sqr[0] );
  polargraph_tree_->index->findNeighbors( knnsearch_result, &cur_describe.globalDescriptor[0] /* query */, nanoflann::SearchParams(bag_search_num) ); 
  std::vector<float> cur_des_local = cur_describe.localDescriptor;
  if(ret.size()<1)
  {
    return false;
  }  
  for (unsigned int i = 0; i < ret.size(); i++)
  {
    index_sort.push_back(ret[i]);
  }
  int min_index = -1;

  for (int i = 0; i < index_sort.size(); i++)
  {
    for(int j = 0; j < index_sort.size() - i; j++)
    {
      int temp_sort;
      if(index_sort[j] > index_sort[j + 1])
      {
        temp_sort = index_sort[j + 1];
        index_sort[j + 1] = index_sort[j];
        index_sort[j] = temp_sort;
      }
    }
  }

  min_index = index_sort[0];
  if(min_index == -1)
  {
    return false;
  }
  // int vector_index = 0;
  // int vector_vector_index = 0;
  // std::vector<std::vector<size_t>> num_vector;
  // num_vector.resize(index_sort.size());
  // num_vector[0].push_back(index_sort[0]);
  // for(int i = 1; i < index_sort.size(); i++)
  // {
  //   if((index_sort[i] - num_vector[vector_index][vector_vector_index]) < 10)
  //   {
  //     num_vector[vector_index].push_back
  //   }
  // }
  double score_th = 0.1;

  int index_num = 0;
  double max_score = -1;
  double max_score1 = 0;
  int RANSAC_num = 0;
  int max_index = 0;
  int inlen_num = 0;
  int match_num = 0;
  MatrixXi inlierID_1;
  MatrixXi matcherID_1;
  MatrixXf R_1;
  MatrixXf T_1;
  R_1.setIdentity();
  T_1.setZero();
  while (index_num < index_sort.size())
  {
    Descriptor last_describe = histoary_describe[index_sort[index_num]];
    std::vector<float> last_des_local = last_describe.localDescriptor;
    MatrixXi matcherID;
    Eigen::Vector3d thred_vector(0.9, 10, 0.95);
    matcher matches(cur_describe, last_describe, 7, thred_vector);
    matcherID = matches.getGoodMatcher(0.3);

    
    double socre1 = (double)(matcherID.rows())/(double)(cur_describe.labelVector.size());
    if(socre1 < 0.35) // kitti 0.4
    {
      index_num =index_num + 1;
      continue;
    }
    if((double)(matcherID.rows()) < 7) // kitti 0.4
    {
      index_num =index_num + 1;
      continue;
    }
    // std::pair<Eigen::Vector3d, Eigen::Vector3d> now_loop_point;
    // now_loop_point.first = cur_pose;
    // now_loop_point.second = histoary_pose[index_sort[index_num]];
    // if((now_loop_point.first - now_loop_point.second).norm() >= 10)
    // {
    //   index_num =index_num + 1;
    //   continue;
    // }
    registration registration(cur_center, histoary_center[index_sort[index_num]], matcherID);
    registration.matcherRANSAC(0.4, 7, 500);
    MatrixXi inlierID;
    inlierID = registration.inlierID;
    double socre = (double)(inlierID.rows() - 1)/(double)(matcherID.rows());
    if(socre > score_th)
    {
      registration.Alignment();
    }
    // int size11 = cur_des_local.size();
    // int size22 = last_des_local.size();
    // float SumTop = 0;
    // float SumBottomLeft = 0; 
    // float SumBottomRight = 0; 
    // int cont = 0;
    // int cont_all = 0;
    // int cont_false = 0;
    // for(int i = 0; i < size11; i++)
    // {
    //   if(cur_des_local[i]!= 0 && last_des_local[i] != 0)
    //   {
    //     if(abs(cur_des_local[i]- last_des_local[i]) < 15)
    //       {cont = cont + 1;}
    //     cont_all = cont_all + 1;
    //   }
    //     // SumBottomLeft = SumBottomLeft + pow(cur_des_local[i], 2);
    //     // SumBottomRight = SumBottomRight + pow(last_des_local[i], 2);
    // }
    // float SumBottom = sqrt(SumBottomLeft*SumBottomRight);
    // float score1 = matches.getGoodscore(0.9);
    if(socre >= max_score)
    {
      max_score = socre;
      max_score1 = socre1;
      max_index = index_sort[index_num];
      match_num = matcherID.rows();
      RANSAC_num = inlierID.rows() - 1;
      matcherID_1 = matcherID;
      inlierID_1 = inlierID;
      if(socre > score_th)
      {
        R_1 = registration.Rotation;
        T_1 = registration.Translation;
      }
    }
    if(socre > score_th/2)
    {
      inlen_num = inlen_num+1;
    }
    // std::pair<Eigen::Vector3d, Eigen::Vector3d> now_loop_point;
    // now_loop_point.first = cur_pose;
    // now_loop_point.second = histoary_pose[index_sort[index_num]];
    // if((now_loop_point.first - now_loop_point.second).norm() < 10)
    // {
    //       *loop_index = index_sort[index_num];
    //       return true;
    // }
    std::cout<< "matcher_num" << (double)(index_sort[index_num]) <<"matcher_score:"<< socre <<"matcher num:"<<(double)(matcherID.rows())<<std::endl;
    index_num =index_num + 1;
  }
   std::cout<< "matcher_ID" << max_index <<"matcher_score:"<< max_score <<std::endl;
  if(max_score > score_th) // kitti00 0.22
  {
    std::cout<< "matcher_ID" << max_index <<"matcher_score:"<< max_score <<std::endl;
    *loop_index = max_index;
    *match_id = matcherID_1;
    *linlier_id = inlierID_1;
    *R = R_1;
    *T = T_1;
    return true;
  }
  else
  {
    return false;
  }
}

void sem_graph::visualizeLoopConstraintEdge(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> loop_frames, double time_stap)
{
    visualization_msgs::MarkerArray markerArray;
    // loop nodes
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "world";
    markerNode.header.stamp = ros::Time().fromSec(time_stap);
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 1; markerNode.scale.y = 1; markerNode.scale.z = 1; 
    markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
    markerNode.color.a = 1;

    // loop edges
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = "world";
    markerEdge.header.stamp = ros::Time().fromSec(time_stap);
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.4; markerEdge.scale.y = 0.4; markerEdge.scale.z = 0.4;
    markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
    markerEdge.color.a = 1;
    for(int i = 0; i < loop_frames.size(); i++)
    {
      Eigen::Vector3d cur_frame = loop_frames[i].first;
      Eigen::Vector3d old_frame = loop_frames[i].second;
      if((cur_frame-old_frame).norm()>=15)
      {
        continue;
      }
      geometry_msgs::Point p;
      p.x = cur_frame.x();
      p.y = cur_frame.y();
      p.z = cur_frame.z();
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
      p.x = old_frame.x();
      p.y = old_frame.y();
      p.z = old_frame.z();
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);

      markerArray.markers.push_back(markerNode);
      markerArray.markers.push_back(markerEdge);
    }
    pubLoopConstraintEdge_.publish(markerArray);
}

visualization_msgs::MarkerArray sem_graph::visualizeMatchID(std::vector<Eigen::Vector4f> cur_center, std::vector<Eigen::Vector4f> last_center, MatrixXi matchid, double time_stap, string pub_name)
{
  visualization_msgs::MarkerArray output;
  visualization_msgs::Marker markerEdge;
  markerEdge.header.frame_id = "base_link";
  markerEdge.header.stamp = ros::Time().fromSec(time_stap);
  markerEdge.action = visualization_msgs::Marker::ADD;
  markerEdge.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge.ns = pub_name;
  markerEdge.id = 1;
  markerEdge.pose.orientation.w = 1;
  markerEdge.scale.x = 0.6; markerEdge.scale.y = 0.6; markerEdge.scale.z = 0.6;
  markerEdge.color.r = 0.447; markerEdge.color.g = 0.623; markerEdge.color.b = 0.812;
  markerEdge.color.a = 0.8;

  int size = matchid.rows();
  for(int i=0; i<size; i++)
  {
        int row1 = matchid(i, 0);
        int row2 = matchid(i, 1);
        geometry_msgs::Point p1, p2;

        p1.x = cur_center[row1][0];
        p1.y = cur_center[row1][1];
        p1.z = cur_center[row1][2];

        p2.x = last_center[row2][0];
        p2.y = last_center[row2][1];
        p2.z = last_center[row2][2]+30;
        markerEdge.points.push_back(p1);
        markerEdge.points.push_back(p2);
        output.markers.push_back(markerEdge);
    }
    return output;
}

visualization_msgs::MarkerArray sem_graph::visualizegraphedge(std::vector<Eigen::Vector4f> cur_center, MatrixXf graph_max, double time_stap)
{
  visualization_msgs::MarkerArray output;
  visualization_msgs::Marker markerEdge;
  markerEdge.header.frame_id = "base_link";
  markerEdge.header.stamp = ros::Time().fromSec(time_stap);
  markerEdge.action = visualization_msgs::Marker::ADD;
  markerEdge.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge.ns = "graph_edge";
  markerEdge.id = 1;
  markerEdge.pose.orientation.w = 1;
  markerEdge.scale.x = 0.4; markerEdge.scale.y = 0.4; markerEdge.scale.z = 0.4;
  markerEdge.color.r = 0.812; markerEdge.color.g = 0.423; markerEdge.color.b = 0.447;
  markerEdge.color.a = 0.8;

  visualization_msgs::Marker markerEdge2;
  markerEdge2.header.frame_id = "base_link";
  markerEdge2.header.stamp = ros::Time().fromSec(time_stap);
  markerEdge2.action = visualization_msgs::Marker::ADD;
  markerEdge2.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge2.ns = "graph_edge2";
  markerEdge2.id = 1;
  markerEdge2.pose.orientation.w = 1;
  markerEdge2.scale.x = 0.4; markerEdge2.scale.y = 0.4; markerEdge2.scale.z = 0.4;
  markerEdge2.color.r = 0.305; markerEdge2.color.g = 0.702; markerEdge2.color.b = 0.03;
  markerEdge2.color.a = 0.8;

  int size = graph_max.rows();
  for(int Num=0; Num<size; Num++)
  {
    for(int NeiNum = Num; NeiNum<size; NeiNum++)
    {
      if(graph_max(Num, NeiNum) != 0 && Num != NeiNum)
      {
        if(graph_max(Num, NeiNum) == 1)
        {
          geometry_msgs::Point p1, p2;

          p1.x = cur_center[Num][0];
          p1.y = cur_center[Num][1];
          p1.z = cur_center[Num][2];

          p2.x = cur_center[NeiNum][0];
          p2.y = cur_center[NeiNum][1];
          p2.z = cur_center[NeiNum][2];
          markerEdge.points.push_back(p1);
          markerEdge.points.push_back(p2);
          output.markers.push_back(markerEdge);
        }
        else if(graph_max(Num, NeiNum) == 2)
        {
          geometry_msgs::Point p1, p2;

          p1.x = cur_center[Num][0];
          p1.y = cur_center[Num][1];
          p1.z = cur_center[Num][2];

          p2.x = cur_center[NeiNum][0];
          p2.y = cur_center[NeiNum][1];
          p2.z = cur_center[NeiNum][2];
          markerEdge2.points.push_back(p1);
          markerEdge2.points.push_back(p2);
          output.markers.push_back(markerEdge2);
        }

      }
    }

  }
    return output;
}

visualization_msgs::MarkerArray sem_graph::visualizegraphedge_last(std::vector<Eigen::Vector4f> cur_center, MatrixXf graph_max, double time_stap)
{
  visualization_msgs::MarkerArray output;
  visualization_msgs::Marker markerEdge;
  markerEdge.header.frame_id = "base_link";
  markerEdge.header.stamp = ros::Time().fromSec(time_stap);
  markerEdge.action = visualization_msgs::Marker::ADD;
  markerEdge.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge.ns = "graph_edge_last";
  markerEdge.id = 1;
  markerEdge.pose.orientation.w = 1;
  markerEdge.scale.x = 0.4; markerEdge.scale.y = 0.4; markerEdge.scale.z = 0.4;
  markerEdge.color.r = 0.812; markerEdge.color.g = 0.423; markerEdge.color.b = 0.447;
  markerEdge.color.a = 0.8;

  visualization_msgs::Marker markerEdge2;
  markerEdge2.header.frame_id = "base_link";
  markerEdge2.header.stamp = ros::Time().fromSec(time_stap);
  markerEdge2.action = visualization_msgs::Marker::ADD;
  markerEdge2.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge2.ns = "graph_edge2_last";
  markerEdge2.id = 1;
  markerEdge2.pose.orientation.w = 1;
  markerEdge2.scale.x = 0.4; markerEdge2.scale.y = 0.4; markerEdge2.scale.z = 0.4;
  markerEdge2.color.r = 0.305; markerEdge2.color.g = 0.702; markerEdge2.color.b = 0.03;
  markerEdge2.color.a = 0.8;

  int size = graph_max.rows();
  for(int Num=0; Num<size; Num++)
  {
    for(int NeiNum = Num; NeiNum<size; NeiNum++)
    {
      if(graph_max(Num, NeiNum) != 0 && Num != NeiNum)
      {
        if(graph_max(Num, NeiNum) == 1)
        {
          geometry_msgs::Point p1, p2;

          p1.x = cur_center[Num][0];
          p1.y = cur_center[Num][1];
          p1.z = cur_center[Num][2]+50;

          p2.x = cur_center[NeiNum][0];
          p2.y = cur_center[NeiNum][1];
          p2.z = cur_center[NeiNum][2]+30;
          markerEdge.points.push_back(p1);
          markerEdge.points.push_back(p2);
          output.markers.push_back(markerEdge);
        }
        else if(graph_max(Num, NeiNum) == 2)
        {
          geometry_msgs::Point p1, p2;

          p1.x = cur_center[Num][0];
          p1.y = cur_center[Num][1];
          p1.z = cur_center[Num][2]+50;

          p2.x = cur_center[NeiNum][0];
          p2.y = cur_center[NeiNum][1];
          p2.z = cur_center[NeiNum][2]+30;
          markerEdge2.points.push_back(p1);
          markerEdge2.points.push_back(p2);
          output.markers.push_back(markerEdge2);
        }

      }
    }

  }
    return output;
}
int main(int argc, char **argv) {
  ros::init(argc, argv, "ros1_demo");
  ros::NodeHandle pnh("~");
  sem_graph node(&pnh);
  ros::spin();
  
  return 0;
}