# TestOpenPose
Interactive OpenPose-related applications.

## Environment

### Basic Environment

1. [Anaconda](https://www.anaconda.com/products/distribution?gclid=Cj0KCQjwyMiTBhDKARIsAAJ-9VulkR13yJuDAfHYao5OeinS8WAIEQhm_AKIJkDC8TAUcLaTWjkTiioaAtd6EALw_wcB)



### Camera SDK

1. [FLIR Camera Spinnaker SDK](https://www.flir.com/products/spinnaker-sdk/?vertical=machine+vision&segment=iis)



### Core Algorithm

1. [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
2. [ProHMR](https://github.com/nkolot/ProHMR)
3. [SPIN](https://github.com/nkolot/SPIN)



### Rendering Engine

1. [Taichi](https://github.com/taichi-dev/taichi)==0.7.13
4. [Tina](https://github.com/taichi-dev/taichi_three)==0.1.1



## Command

### ProHMR

#### Taichi Backend

```shell
python ProHMR_taichi_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --cam=https://10.0.0.139:8080/video --screen_width=800 --debug_performance=0

python ProHMR_taichi_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --run_fitting --cam=https://10.0.0.139:8080/video --screen_width=800 --debug_performance=0
```



#### 

```shell
python ProHMR_taichi_spinnaker_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --screen_width=800 --debug_performance=0

python ProHMR_taichi_spinnaker_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --run_fitting --screen_width=800 --debug_performance=0
```





#### Pyrender Offline Backend

```shell
python ProHMR_demo_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --cam=https://10.0.0.139:8080/video --screen_width=800 --run_open_pose

python ProHMR_demo_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --run_fitting --cam=https://10.0.0.139:8080/video --screen_width=800 --run_open_pose
```





```shell
python ProHMR_demo_spinnaker_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --out_folder=out --screen_width=800 --run_open_pose

python ProHMR_demo_spinnaker_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --out_folder=out --run_fitting --screen_width=800 --run_open_pose
```



### SPIN

```shell
python SPIN_demo_camera.py --checkpoint=data/model_checkpoint.pt --cam=https://10.0.0.139:8080/video --screen_width=800
```

```shell
python SPIN_demo_spinnaker_camera.py --checkpoint=data/model_checkpoint.pt --screen_width=800
```



## Deprecated Command

```shell
python demo_camera.py --img_folder=ProHMR/ProHMR/example_data/images --keypoint_folder=ProHMR/ProHMR/example_data/keypoints --checkpoint=ProHMR/ProHMR/data/checkpoint.pt --model_cfg=ProHMR/ProHMR/prohmr/configs/prohmr.yaml --out_folder=out --run_fitting



python demo_camera.py --img_folder=ProHMR/ProHMR/example_data/images --keypoint_folder=ProHMR/ProHMR/example_data/keypoints --checkpoint=ProHMR/ProHMR/data/checkpoint.pt --out_folder=out --run_fitting



python demo_camera.py --img_folder=ProHMR/ProHMR/example_data/images --keypoint_folder=ProHMR/ProHMR/example_data/keypoints --out_folder=out --run_fitting
```

