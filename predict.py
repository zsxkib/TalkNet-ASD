# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import shutil
import subprocess
from typing import List
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(
        self,
        video: Path = Input(description="Path to the video"),
        face_det_scale: float = Input(default=0.25, description="Scale factor for face detection, the frames will be scaled to 0.25 of the original", ge=0, le=1),
        min_track: int = Input(default=10, description="Number of min frames for each shot"),
        num_failed_det: int = Input(default=10, description="Number of missed detections allowed before tracking is stopped", ge=1),
        min_face_size: int = Input(default=1, description="Minimum face size in pixels", ge=1),
        crop_scale: float = Input(default=0.40, description="Scale bounding box", ge=0, le=1),
        start: int = Input(default=0, description="The start time of the video", ge=0),
        duration: int = Input(default=-1, description="The duration of the video, when set as -1, will extract the whole video"),
    ) -> List[Path]:
        video_path = str(video)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_folder = "demo"
        
        shutil.rmtree(video_folder, ignore_errors=True)
        os.makedirs(video_folder, exist_ok=True)

        target_video_path = os.path.join(video_folder, os.path.basename(video_path))
        shutil.copy(video_path, target_video_path)

        duration = max(0, duration)
        n_data_loader_thread = 32
        command = f"python demoTalkNet.py --videoName {video_name} " \
                  f"--videoFolder {video_folder} " \
                  f"--pretrainModel pretrain_TalkSet.model " \
                  f"--nDataLoaderThread {n_data_loader_thread} " \
                  f"--facedetScale {face_det_scale} " \
                  f"--minTrack {min_track} " \
                  f"--numFailedDet {num_failed_det} " \
                  f"--minFaceSize {min_face_size} " \
                  f"--cropScale {crop_scale} " \
                  f"--start {start} " \
                  f"--duration {duration} "

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(f"Command output: {stdout.decode()}")
        if stderr:
            print(f"Command errors: {stderr.decode()}")

        mp4_files = []
        excluded_files = ["video_only.avi", "video.avi"]
        avi_files = [avi_file for avi_file in Path(video_folder).rglob("*.avi") if avi_file.name not in excluded_files]
        for avi_file in avi_files:
            mp4_file = avi_file.with_suffix('.mp4')
            conversion_command = f"ffmpeg -i {avi_file} {mp4_file}"
            conversion_process = subprocess.run(conversion_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,)
            if conversion_process.returncode == 0:
                mp4_files.append(Path(mp4_file))
        return mp4_files