# TestOpenPose
Interactive OpenPose-related applications.



## Command

```shell
python demo_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --out_folder=out --cam=https://10.0.0.139:8080/video --screen_width=800 --run_open_pose

python demo_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --out_folder=out --run_fitting --cam=https://10.0.0.139:8080/video --screen_width=800 --run_open_pose
```

```shell
python demo_spinnaker_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --out_folder=out --screen_width=800 --run_open_pose

python demo_spinnaker_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --out_folder=out --run_fitting --screen_width=800 --run_open_pose
```





## Deprecated Command

```shell
python demo_camera.py --img_folder=ProHMR/ProHMR/example_data/images --keypoint_folder=ProHMR/ProHMR/example_data/keypoints --checkpoint=ProHMR/ProHMR/data/checkpoint.pt --model_cfg=ProHMR/ProHMR/prohmr/configs/prohmr.yaml --out_folder=out --run_fitting



python demo_camera.py --img_folder=ProHMR/ProHMR/example_data/images --keypoint_folder=ProHMR/ProHMR/example_data/keypoints --checkpoint=ProHMR/ProHMR/data/checkpoint.pt --out_folder=out --run_fitting



python demo_camera.py --img_folder=ProHMR/ProHMR/example_data/images --keypoint_folder=ProHMR/ProHMR/example_data/keypoints --out_folder=out --run_fitting
```

