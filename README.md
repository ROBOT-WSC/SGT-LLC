# <div align = "center">SGT-LLC:</div>

## <div align = "center">LiDAR Loop Closing Based on Semantic Graph with Triangular Spatial Topology</div>


> Shaocong Wang, Fengkui Cao, Ting Wang Xieyuanli Chen, Shiliang Shao
>
> [IEEE Robotics and Automation Letters](https://ieeexplore.ieee.org/document/10891171)

## News



* **`1 March 2025`:**  Code updata
* **`10 February2025`:** Accepted by [IEEE RAL](https://ieeexplore.ieee.org/document/10891171)! 

## Getting Started


### Instructions
The real-time demo of SGT-LLC was built on [rangenet-ros](https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/tree/main) . Please refer to [rangenet-ros](https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/tree/main) to configure the semantic segmentation environment.

### Dependencies

- Ubuntu 20.04
- ROS Noetic (`roscpp`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, `pcl_ros`)
- C++ 14
- OpenMP
- Point Cloud Library
- Eigen >=3.3.4
- DBow3

### Compiling

Create a catkin workspace, clone the `SGT-LLC`  repository into the `src` folder, and download the test [semantic segment model]() into `./SGT-LLC/model/`. Finally, Compile via the [`catkin_tools`](https://catkin-tools.readthedocs.io/en/latest/) package:

```sh
mkdir ws && cd ws && mkdir src && catkin init && cd src
git clone https://github.com/ROBOT-WSC/SGT-LLC.git
catkin build
```

### Execution

For your convenience, KITTI, [MulRan](https://sites.google.com/view/mulran-pr/home) and [MCD](https://mcdviral.github.io/)can be real-time test on SGT-LLC. For KITTI, the raw  `rosbag` can be directly used for loop close detection. For MulRan and MCD, we provide some bags with semantic labels. These examples can be found [here](https://drive.google.com/drive/folders/1bt9vWPVgTF8I8JXSUO-Dpi3n2vomG6t9). To run, first launch demo via:

```sh
roslaunch rangenet_pp "kitti/mulrun/mcd".launch
```

In a separate terminal session, play back the downloaded bag:

```sh
rosbag play "bag's name" --clock
```

## Citation

If you find SGT-LLC is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@ARTICLE{10891171,
  author={Wang, Shaocong and Cao, Fengkui and Wang, Ting and Chen, Xieyuanli and Shao, Shiliang},
  journal={IEEE Robotics and Automation Letters}, 
  title={SGT-LLC: LiDAR Loop Closing Based on Semantic Graph With Triangular Spatial Topology}, 
  year={2025},
  volume={10},
  number={4},
  pages={3326-3333},
  keywords={Semantics;Topology;Point cloud compression;6-DOF;Data mining;Laser radar;Optimization;Encoding;Accuracy;Vectors;SLAM;localization;mapping},
  doi={10.1109/LRA.2025.3542695}}
```
## Acknowledgements

We thank the authors of the [semantic-histogram-based-global-localization](https://github.com/gxytcrc/semantic-histogram-based-global-localization?tab=readme-ov-file) and [rangenet-ros](https://github.com/Natsu-Akatsuki/RangeNet-TensorRT/tree/main) open-source package.
