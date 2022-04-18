# TestOpenPose
Test OpenPose application in real time.



## Command

```shell
python demo_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --out_folder=out --run_open_pose --cam=https://10.0.0.139:8080/video

python demo_camera.py --img_folder=example_data/images --keypoint_folder=example_data/keypoints --out_folder=out --run_fitting --cam=https://10.0.0.139:8080/video
```



## Deprecated Command

```shell
python demo_camera.py --img_folder=ProHMR/ProHMR/example_data/images --keypoint_folder=ProHMR/ProHMR/example_data/keypoints --checkpoint=ProHMR/ProHMR/data/checkpoint.pt --model_cfg=ProHMR/ProHMR/prohmr/configs/prohmr.yaml --out_folder=out --run_fitting



python demo_camera.py --img_folder=ProHMR/ProHMR/example_data/images --keypoint_folder=ProHMR/ProHMR/example_data/keypoints --checkpoint=ProHMR/ProHMR/data/checkpoint.pt --out_folder=out --run_fitting



python demo_camera.py --img_folder=ProHMR/ProHMR/example_data/images --keypoint_folder=ProHMR/ProHMR/example_data/keypoints --out_folder=out --run_fitting
```

