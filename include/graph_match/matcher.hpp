#ifndef matcher_hpp
#define matcher_hpp

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include <iostream>  
#include <stdio.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>
#include "neighborhood.hpp"


using namespace cv;
using namespace std;
using namespace Eigen;

class matcher
{
    public:
        matcher(Descriptor Ds1, Descriptor Ds2);
        matcher(Descriptor Ds1, Descriptor Ds2, int type); 
        matcher(Descriptor Ds1, Descriptor Ds2, int label_size, Eigen::Vector3d thred_local);
        ~matcher();
        MatrixXf scoreMatrix;
        VectorXf last_picked;
        MatrixXi getMatcherID();
        MatrixXi getGoodMatcher();
        MatrixXi getGoodMatcher(float global_thred);
        float getGoodscore(float global_thred);
        int matcher_num;

    private:
        int size1;
        int size2;
        VectorXf noNei;
};

#endif /* pointCloudMapping */

