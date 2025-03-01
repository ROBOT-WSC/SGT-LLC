#ifndef neighborhood_hpp
#define neighborhood_hpp

#include <iostream>  
#include <stdio.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>
#include "opencv2/core.hpp"


#include <cstdlib>
#include <ctime>


using namespace std;
using namespace Eigen;

class Neighborhood
{
    public:
        Neighborhood(vector<Vector4f> Cpoint);
        ~Neighborhood();
        std::pair<MatrixXf, MatrixXf> getNeighbor();
        VectorXf label;
        vector<vector<float> > centerpoint;
        
        
};

class Descriptor
{
    public:
        Descriptor(Neighborhood Nei, int stepNumber, int stepLengh);
        Descriptor(Neighborhood Nei);
        Descriptor(Neighborhood Nei, int Histogram);
        Descriptor(Neighborhood Nei, int graph_depth, int label_score, int geometry_type);
        ~Descriptor();
        MatrixXf getDescriptor(int DesID);
        MatrixXf vis_edge;
        cv::Mat getDescriptor();
        int size();
        int SL;
        int SN;
        VectorXf noNeighbor;
        std::vector<float> globalDescriptor;
        std::vector<float> localDescriptor;
        std::vector<float> localDescriptor_cos;
        std::vector<float> localDescriptor_dis;
        VectorXf labelVector;

    private:
        vector<MatrixXf> DescriptorVector;
        MatrixXf neighbor;
        MatrixXf dis_mat;
        std::pair<MatrixXf, MatrixXf> neighbor_dis;
};



#endif /* pointCloudMapping */
