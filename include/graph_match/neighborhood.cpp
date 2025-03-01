#include "neighborhood.hpp"

Neighborhood::Neighborhood(vector<Vector4f> Cpoint){
    // cout<<"Neighborhood construct"<<endl;
    int size = Cpoint.size();
    label.resize(size);
    for(int i =0; i<size; i++){
        vector<float> cp(4);
        cp[0] = Cpoint[i][0]; cp[1] = Cpoint[i][1]; cp[2] = Cpoint[i][2]; cp[3] = Cpoint[i][3]; 
        label(i) = cp[3];
        centerpoint.push_back(cp);
    }
}

Neighborhood::~Neighborhood(){
    // cout<<"The Neighborhood object destroyed"<<endl;
}
//范围内全连接
std::pair<MatrixXf, MatrixXf> Neighborhood::getNeighbor(){

    MatrixXf neighbor(centerpoint.size(), centerpoint.size());
    MatrixXf dis_mat(centerpoint.size(), centerpoint.size());
    neighbor.setZero();
    dis_mat.setZero();
    for(int n = 0; n < centerpoint.size(); n++){
        float x0 = centerpoint[n][0];
        float y0 = centerpoint[n][1];
        float z0 = centerpoint[n][2];
        for(int m = n+1; m<centerpoint.size(); m++){
            float x1 = centerpoint[m][0];
            float y1 = centerpoint[m][1];
            float z1 = centerpoint[m][2];
            float distance = sqrt(pow(x0-x1,2)+ pow(y0-y1,2) + pow(z0-z1,2));
            // cout<<"label0: "<<centerpoint[n][3]<<endl;
            // cout<<"label1: "<<centerpoint[m][3]<<endl;
            // cout<<"distance: "<<distance<<endl;

            if(distance<30){
                neighbor(m, n) = 1;
                neighbor(n, m) = 1;
                dis_mat(m, n) = distance;
                dis_mat(n, m) = distance;

            }
        }
    }
    std::pair<MatrixXf, MatrixXf> neighbor_dis;
    neighbor_dis.first = neighbor;
    neighbor_dis.second = dis_mat;
    return neighbor_dis;

}

//-----------------------------------------------------------------------------------------------
// Random walk descriptor
Descriptor::Descriptor(Neighborhood Nei, int stepNumber, int stepLengh){
    clock_t startTime,endTime;
    startTime = clock();
    cout<<"Descriptor construct"<<endl;
    //obtain the neibore relations;
    neighbor_dis = Nei.getNeighbor();
    neighbor = neighbor_dis.first;
    dis_mat = neighbor_dis.second;
    labelVector = Nei.label;
    cout<<"vector size: "<<labelVector.size()<<endl;
    //cout<<labelVector<<endl;
    //inite the no neighbor vector;
    noNeighbor.resize(labelVector.size());
    noNeighbor.setZero();
    SN = stepNumber;
    SL = stepLengh;
    endTime = clock();
    int timedifference = (int)((endTime - startTime)*10000);
    int size = neighbor.rows();
    cout<<"time1: "<<time(0)<<endl;
    srand((int)time(0)+timedifference);
    cout<<"test1"<<endl;
    for(int row = 0; row<size; row++){

        MatrixXf SingleDescriptor(SN, SL);
        float label;
        float PointID;
        for(int stepNum = 0; stepNum<SN; stepNum++){
            for(int stepDepth = 0; stepDepth<SL; stepDepth++){
                if(stepDepth == 0){
                    PointID = row;
                }
                label = labelVector(PointID);
                SingleDescriptor(stepNum, stepDepth) = label;
                
                //find the neighbor point
                int count = 0;
                vector<int> neighborID;
                for(int IDnum = 0; IDnum<size; IDnum++){
                    if(neighbor(PointID, IDnum) == 1){
                        neighborID.push_back(IDnum);
                        count = count+1;
                    }
                }
                if(count == 0){
                    noNeighbor(PointID) = 1;
                    continue;
                }
                int pathID = rand()%count;
                PointID = neighborID[pathID];
            }
        }

        DescriptorVector.push_back(SingleDescriptor);
    }


}
//FAST Descriptor
Descriptor::Descriptor(Neighborhood Nei){
    cout<<"Descriptor construct"<<endl;
    neighbor_dis = Nei.getNeighbor();
    neighbor = neighbor_dis.first;
    dis_mat = neighbor_dis.second;
    labelVector = Nei.label;
    //inite the no neighbor vector;
    noNeighbor.resize(labelVector.size());
    noNeighbor.setZero();
    int size = neighbor.rows();
    int labelSize = 7;
    for(int Num =0; Num<size; Num++){
        MatrixXf SingleDescriptor(labelSize, 1);
        SingleDescriptor.setZero();
        //colmum 2
        //find the neibor point
        int neiCount = 0;
        vector<int> neighborID;
        for(int NeiNum = 0; NeiNum<size; NeiNum++){
            if(neighbor(Num, NeiNum) == 1){
                int locallabel = labelVector(NeiNum);
                SingleDescriptor(locallabel, 0) = SingleDescriptor(locallabel, 0) + 1;
                neighborID.push_back(NeiNum);
                neiCount = neiCount+1;
            }
        }
        if(neiCount ==0){
            noNeighbor(Num) = 1;
            DescriptorVector.push_back(SingleDescriptor);
            continue;
        }
        DescriptorVector.push_back(SingleDescriptor);
    }
}

Descriptor::Descriptor(Neighborhood Nei, int Histogram){
    cout<<"Descriptor construct"<<endl;
    neighbor_dis = Nei.getNeighbor();
    neighbor = neighbor_dis.first;
    dis_mat = neighbor_dis.second;

    labelVector = Nei.label;
    //inite the no neighbor vector;
    noNeighbor.resize(labelVector.size());
    noNeighbor.setZero();
    int size = neighbor.rows();
    int labelSize = 7;
    for(int Num =0; Num<size; Num++){
        MatrixXf SingleDescriptor(343, 1);
        SingleDescriptor.setZero();
        int VectorNum = 0;
        //colmum 2
        //find the neibor point
        int neiCount = 0;
        vector<int> neighborID;
        for(int NeiNum = 0; NeiNum<size; NeiNum++){
            if(neighbor(Num, NeiNum) == 1){
                neighborID.push_back(NeiNum);
                neiCount = neiCount+1;
            }
        }
        if(neiCount ==0){
            noNeighbor(Num) = 1;
            DescriptorVector.push_back(SingleDescriptor);
            continue;
        }
        //colmum 3
        int pointlable = labelVector(Num);
        for(int NeiID =0; NeiID<neiCount; NeiID++){

            int NeiNum = neighborID[NeiID];
            int previoudlable = labelVector(NeiNum);

            for(int NeiNeiNum = 0; NeiNeiNum<size; NeiNeiNum++){
                if(neighbor(NeiNum, NeiNeiNum) == 1){
                    int locallabel = labelVector(NeiNeiNum);
                    int VectorNum = pointlable * 49 + previoudlable*7 + locallabel;
                    SingleDescriptor(VectorNum, 0) = SingleDescriptor(VectorNum, 0) + 1;
                }
            }
        }
        DescriptorVector.push_back(SingleDescriptor);
    }
}
//geometry improve
Descriptor::Descriptor(Neighborhood Nei, int graph_depth, int label_score, int geometry){
    // cout<<"My Descriptor construct"<<endl;
    vis_edge.resize(Nei.centerpoint.size(), Nei.centerpoint.size());
    vis_edge.setZero();
    neighbor_dis = Nei.getNeighbor();
    neighbor = neighbor_dis.first;
    dis_mat = neighbor_dis.second;
    labelVector = Nei.label;
    //inite the no neighbor vector;
    noNeighbor.resize(labelVector.size());
    noNeighbor.setZero();
    int size = neighbor.rows();
    int labelSize = 7;
    globalDescriptor.resize(labelSize*labelSize*labelSize);
    for(int z = 0; z < globalDescriptor.size(); z++)
    {
        globalDescriptor[z] = 0;
    }
    //dot
    localDescriptor.resize(labelSize*labelSize*labelSize);
    for(int z = 0; z < localDescriptor.size(); z++)
    {
        localDescriptor[z] = 0;
    }
    //cosin
    localDescriptor_cos.resize(labelSize*labelSize*labelSize);
    for(int z = 0; z < localDescriptor_cos.size(); z++)
    {
        localDescriptor_cos[z] = 1;
    }
    //distance
    localDescriptor_dis.resize(labelSize*labelSize*labelSize);
    for(int z = 0; z < localDescriptor_dis.size(); z++)
    {
        localDescriptor_dis[z] = 0;
    }

    for(int Num =0; Num<size; Num++){
        //一元边
        VectorXf first_max_index(labelSize);
        VectorXf first_min_index(labelSize);
        VectorXf first_max(labelSize);
        VectorXf first_min(labelSize);
        first_max_index.setZero();
        first_min_index.setZero();
        first_max.setZero();
        first_min.setZero();
        //二元边
        VectorXf second_max_index(labelSize);
        VectorXf second_min_index(labelSize);
        VectorXf second_max(labelSize);
        VectorXf second_min(labelSize); 
        second_max_index.setZero();
        second_min_index.setZero();
        second_max.setZero();
        second_min.setZero();
        //全边
        VectorXf both_first_index(labelSize*labelSize);
        VectorXf both_second_index(labelSize*labelSize);
        VectorXf both_max(labelSize*labelSize);
        both_first_index.setZero();
        both_second_index.setZero();
        both_max.setZero();

        MatrixXf SingleDescriptor(labelSize*labelSize*3 + 1, 1);
        
        SingleDescriptor.setZero();
        
        SingleDescriptor(0, 0) = label_score * (labelVector(Num) + 1);
        
        int VectorNum = 0;
        //colmum 2
        //find the neibor point
        int neiCount = 0;
        vector<int> neighborID;
        for(int NeiNum = 0; NeiNum<size; NeiNum++){
            if(neighbor(Num, NeiNum) == 1 && Num != NeiNum){
                neighborID.push_back(NeiNum);
                if(neiCount == 0)
                {
                    first_max(labelVector(NeiNum)) = dis_mat(Num, NeiNum);
                    first_min(labelVector(NeiNum)) = dis_mat(Num, NeiNum);
                    first_max_index(labelVector(NeiNum)) = NeiNum;
                    first_min_index(labelVector(NeiNum)) = NeiNum;
                }
                else
                {
                    if(dis_mat(Num, NeiNum) > first_max(labelVector(NeiNum)))
                    {
                        first_max(labelVector(NeiNum)) = dis_mat(Num, NeiNum);
                        first_max_index(labelVector(NeiNum)) = NeiNum;
                    }
                    if(dis_mat(Num, NeiNum) < first_min(labelVector(NeiNum)))
                    {
                        first_min(labelVector(NeiNum)) = dis_mat(Num, NeiNum);
                        first_min_index(labelVector(NeiNum)) = NeiNum;
                    }
                }
                neiCount = neiCount+1;
            }
        }
        if(neiCount ==0){
            noNeighbor(Num) = 1;
            DescriptorVector.push_back(SingleDescriptor);
            continue;
        }
        //colmum 3
        int neineiCount = 0;
        int pointlable = labelVector(Num);
        for(int NeiID =0; NeiID<neiCount; NeiID++){

            int NeiNum = neighborID[NeiID];
            int previoudlable = labelVector(NeiNum);

            for(int NeiNeiNum = 0; NeiNeiNum<size; NeiNeiNum++){
                if(neighbor(NeiNum, NeiNeiNum) == 1 && NeiNum != NeiNeiNum){  
                    int locallabel = labelVector(NeiNeiNum);
                    if(neineiCount == 0)
                    {
                        //二级节点的边
                        second_max(locallabel) = dis_mat(Num, NeiNum) + dis_mat(NeiNum, NeiNeiNum);
                        second_min(locallabel) = dis_mat(Num, NeiNum) + dis_mat(NeiNum, NeiNeiNum);
                        second_max_index(locallabel) = NeiNeiNum;
                        second_min_index(locallabel) = NeiNeiNum;
                        //二元边
                        both_max(previoudlable*labelSize + locallabel) = dis_mat(Num, NeiNum) + dis_mat(NeiNum, NeiNeiNum);
                        both_first_index(previoudlable*labelSize + locallabel) = NeiNum;
                        both_second_index(previoudlable*labelSize + locallabel) = NeiNeiNum;       
                    }
                    else
                    {
                        if((dis_mat(Num, NeiNum) + dis_mat(NeiNum, NeiNeiNum)) > second_max(locallabel))
                        {
                            second_max(locallabel) = dis_mat(Num, NeiNum) + dis_mat(NeiNum, NeiNeiNum);
                            second_max_index(locallabel) = NeiNeiNum;
                        }
                        if((dis_mat(Num, NeiNum) + dis_mat(NeiNum, NeiNeiNum)) < second_min(locallabel))
                        {
                            second_min(locallabel) = dis_mat(Num, NeiNum) + dis_mat(NeiNum, NeiNeiNum);
                            second_min_index(locallabel) = NeiNeiNum;
                        }
                        if((dis_mat(Num, NeiNum) + dis_mat(NeiNum, NeiNeiNum)) > both_max(previoudlable*labelSize + locallabel))
                        {
                            both_max(previoudlable*labelSize + locallabel) = dis_mat(Num, NeiNum) + dis_mat(NeiNum, NeiNeiNum);
                            both_first_index(previoudlable*labelSize + locallabel) = NeiNum;
                            both_second_index(previoudlable*labelSize + locallabel) = NeiNeiNum; 
                        }
                    }
                    neineiCount = neineiCount + 1;
                }
            }
        }
        //colmum 4
        for(int desc_label1 = 0; desc_label1 < labelSize; desc_label1++)
        {
            for(int desc_label2 = 0; desc_label2 < labelSize; desc_label2++)
            {
                    Eigen::Vector3d des_vector;

                    Eigen::Vector3d des_vector_cos;

                    Eigen::Vector3d des_vector_dis;

                    Eigen::Vector3d cur_point(Nei.centerpoint[Num][0], Nei.centerpoint[Num][1], Nei.centerpoint[Num][2]);
                    
                    Eigen::Vector3d first_point_max(Nei.centerpoint[first_max_index[desc_label1]][0], Nei.centerpoint[first_max_index[desc_label1]][1], Nei.centerpoint[first_max_index[desc_label1]][2]);
                    Eigen::Vector3d first_point_min(Nei.centerpoint[first_min_index[desc_label1]][0], Nei.centerpoint[first_min_index[desc_label1]][1], Nei.centerpoint[first_min_index[desc_label1]][2]);
                    
                    Eigen::Vector3d second_point_max(Nei.centerpoint[second_max_index[desc_label2]][0], Nei.centerpoint[second_max_index[desc_label2]][1], Nei.centerpoint[second_max_index[desc_label2]][2]);
                    Eigen::Vector3d second_point_min(Nei.centerpoint[second_min_index[desc_label2]][0], Nei.centerpoint[second_min_index[desc_label2]][1], Nei.centerpoint[second_min_index[desc_label2]][2]);
                    
                    Eigen::Vector3d both_point_first(Nei.centerpoint[both_first_index(desc_label1*labelSize + desc_label2)][0], Nei.centerpoint[both_first_index(desc_label1*labelSize + desc_label2)][1], Nei.centerpoint[both_first_index(desc_label1*labelSize + desc_label2)][2]);
                    Eigen::Vector3d both_point_second(Nei.centerpoint[both_second_index(desc_label1*labelSize + desc_label2)][0], Nei.centerpoint[both_second_index(desc_label1*labelSize + desc_label2)][1], Nei.centerpoint[both_second_index(desc_label1*labelSize + desc_label2)][2]);

                    if(vis_edge(Num, both_first_index(desc_label1*labelSize + desc_label2))==0)
                    {
                        vis_edge(Num, both_first_index(desc_label1*labelSize + desc_label2)) = 1;
                    }
                    
                    if(vis_edge(both_first_index(desc_label1*labelSize + desc_label2), both_second_index(desc_label1*labelSize + desc_label2))==0)
                    {
                        vis_edge(both_first_index(desc_label1*labelSize + desc_label2), both_second_index(desc_label1*labelSize + desc_label2)) = 2;
                    }
                
                    des_vector(0) = (both_point_first - cur_point).dot(both_point_second - cur_point);
                    des_vector_cos(0) = 1 - ((both_point_first - cur_point).dot(both_point_second - cur_point))/((both_point_first - cur_point).norm() * (both_point_second - cur_point).norm());//888
                    des_vector_dis(0) = (both_point_first - cur_point).norm() - (both_point_second - cur_point).norm();

                    des_vector(1) = (both_point_first - both_point_second).dot(cur_point - both_point_second);
                    des_vector_cos(1) = 1 - ((both_point_first - both_point_second).dot(cur_point - both_point_second))/((both_point_first - both_point_second).norm() * (cur_point - both_point_second).norm());//893
                    des_vector_dis(1) = (both_point_first - both_point_second).norm() - (cur_point - both_point_second).norm();

                    des_vector(2) = (both_point_first - cur_point).dot(both_point_second - both_point_first);
                    des_vector_cos(2) = 1 - ((both_point_first - cur_point).dot(both_point_second - both_point_first))/((both_point_first - cur_point).norm() * (both_point_second - both_point_first).norm());//
                    des_vector_dis(2) = (both_point_first - cur_point).norm() + (both_point_second - both_point_first).norm();

                    // des_vector(0) = (first_point_max - cur_point).norm() * (first_point_min - cur_point).norm() - (first_point_max - cur_point).dot(first_point_min - cur_point);
                    // des_vector_cos(0) = 1 - ((first_point_max - cur_point).dot(first_point_min - cur_point))/((first_point_max - cur_point).norm() * (first_point_min - cur_point).norm());
                    // des_vector_dis(0) = (first_point_max - cur_point).norm() - (first_point_min - cur_point).norm();

                    // des_vector(1) = (second_point_max - cur_point).norm() * (second_point_min - cur_point).norm() - (second_point_max - cur_point).dot(second_point_min - cur_point);
                    // des_vector_cos(1) = 1 - ((second_point_max - cur_point).dot(second_point_min - cur_point))/((second_point_max - cur_point).norm() * (second_point_min - cur_point).norm());
                    // des_vector_dis(1) = (second_point_max - cur_point).norm() - (second_point_min - cur_point).norm();


                    // des_vector(2) = (both_point_first - cur_point).norm() * (both_point_second - cur_point).norm() - (both_point_first - cur_point).dot(both_point_second - cur_point);
                    // des_vector_cos(2) = 1 - ((both_point_first - cur_point).dot(both_point_second - cur_point))/((both_point_first - cur_point).norm() * (both_point_second - cur_point).norm());
                    // des_vector_dis(2) = (both_point_first - cur_point).norm() + (both_point_second - cur_point).norm();

                    if(first_max[desc_label1] == first_min[desc_label1])
                    {
                        des_vector(0) = 0;
                        des_vector_cos(0) = 0;
                        des_vector_dis(0) = 0;
                    }
                    if(second_max[desc_label2] == second_min[desc_label2])
                    {
                        des_vector(1) = 0;
                        des_vector_cos(1) = 0;
                        des_vector_dis(1) = 0;
                    }
                    if(both_max[desc_label1*labelSize + desc_label2] == 0)
                    {
                        des_vector(2) = 0;
                        des_vector_cos(2) = 0;
                        des_vector_dis(2) = 0;
                    }    
                    int mohulable =0;
                    if(desc_label1 <= 2)
                    {
                        mohulable = 0;
                    }
                    else
                    {
                        mohulable = 1;
                    }
                    int globalNum = pointlable*labelSize*labelSize + desc_label1*labelSize + desc_label2;
                    if(des_vector_cos(1)>globalDescriptor[globalNum])
                    {
                        globalDescriptor[globalNum] = des_vector_cos(1);
                    }
                    int localNum =  pointlable*labelSize*labelSize + desc_label1*labelSize + desc_label2;       
                    if(des_vector[2]>localDescriptor[localNum])
                    {
                        localDescriptor[localNum] = des_vector[2];
                    }  

                    if(des_vector_cos[2]<localDescriptor_cos[localNum])
                    {
                        localDescriptor_cos[localNum] = des_vector_cos[2];
                    }  

                    if(des_vector_dis[2]>localDescriptor_dis[localNum])
                    {
                        localDescriptor_dis[localNum] = des_vector_dis[2];
                    }    
                    for(int desc_channel = 0; desc_channel < 3; desc_channel++)
                    {
                        int VectorNum = desc_label1 * labelSize * 3 + desc_label2*3 + desc_channel + 1;
                        SingleDescriptor(VectorNum, 0) = des_vector(desc_channel);
                    }

            }
        }
        DescriptorVector.push_back(SingleDescriptor);

    }
}

Descriptor::~Descriptor(){
    // cout<<"The Descriptor object destroyed"<<endl;
}

MatrixXf Descriptor::getDescriptor(int DesID){
    MatrixXf SingleDescriptor;
    SingleDescriptor = DescriptorVector[0];
    int rows =  SingleDescriptor.rows();
    int cols = SingleDescriptor.cols();
    
    int des_size = DescriptorVector.size();
    //cout<<"des size: "<<des_size<<endl;
    for(int i = 0; i<des_size; i++){
        if(i == DesID){
            return DescriptorVector[i];
        }
    }
    cout<<"error, no this ID's Descriptor"<<endl;
    MatrixXf errorMat(rows,cols);
    errorMat.setZero();
    return errorMat;
}

cv::Mat Descriptor::getDescriptor(){
    MatrixXf SingleDescriptor;
    SingleDescriptor = DescriptorVector[0];
    int rows =  DescriptorVector.size();
    int cols = SingleDescriptor.rows();
    int des_size = DescriptorVector.size();
    // std::cout<<"rows:"<< rows << "cols:" << cols << std::endl;
    cv::Mat descriptor_mat;
    descriptor_mat = cv::Mat::zeros(rows, cols, CV_32F);
    //cout<<"des size: "<<des_size<<endl;
    for(int i = 0; i < rows; i++)
    {
       for(int j = 0; j < cols; j++)
       {
        descriptor_mat.at<float>(i, j) = (DescriptorVector[i])(j, 0);
       }
    }
    return descriptor_mat.clone();
}

int Descriptor::size(){
    int des_size = DescriptorVector.size();
    return des_size;
}
