import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import sys
import math
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('shank_name.pt')

pipeline = rs.pipeline()
config = rs.config()
wid = 1280
hei = 720
config.enable_stream(rs.stream.color, wid, hei, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, wid, hei, rs.format.z16, 30)
profile = pipeline.start(config)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

depth_scale = 0.0010000000474974513

while True:

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    depth_image = depth_image * depth_scale

    results = model(color_image, conf=0.1)

    obbs = (
        results[0].obb.xywhr.cpu().tolist()
    )

    for i, obb in enumerate(obbs):
        x, y, w, h, angle = obb
        angle_rad = np.deg2rad(angle)
        angle_deg = np.degrees(angle_rad) % 360
        angle_real = ((angle_deg * 18000 / 160 - 180) / 2)

        object_depth = np.median(depth_image[int(y), int(x)]) * 1000
        Xtarget = object_depth * (x - intr.ppx) / intr.fx
        Ytarget = object_depth * (y - intr.ppy) / intr.fy

        dx = w / 2 * np.cos(angle_deg)
        dy = w / 2 * np.sin(angle_deg)
        dhx = h / 2 * np.sin(angle_deg)
        dhy = h / 2 * np.cos(angle_deg)

        corners = [
            (int(x - dx - dhx), int(y - dy + dhy)),
            (int(x + dx - dhx), int(y + dy + dhy)),
            (int(x + dx + dhx), int(y + dy - dhy)),
            (int(x - dx + dhx), int(y - dy - dhy)),
        ]

        label1 = f"{object_depth:.0f} mm"
        coord_string = f"X,Y: {Xtarget:.0f}" + f",{Ytarget:.0f}"
        label2 = f"{angle_real:.0f} degrees"
        """
        x1 = wid * 0.3
        y1 = hei * 0.3
        x2 = wid * 0.7
        y2 = hei * 0.7
        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (30, 30, 200), 2)
        if x1 < x < x2 and y1 < y < y2:
        """
        for ii in range(4):
            cv2.line(
                color_image, corners[ii], corners[(ii + 1) % 4], (252, 119, 30), 2
            )
            cv2.circle(color_image, (int(x), int(y)), 1, (0, 0, 254), 2)
            # Draw the bounding box
            cv2.putText(color_image, label1, (int(x - 14), int(y) - 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (252, 119, 30),
                        1)
            cv2.putText(color_image, label2, (int(x - 14), int(y) - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (252, 119, 30),
                        1)
            cv2.putText(color_image, coord_string, (int(x - 14), int(y) - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (252, 119, 30), 1)


    if (cv2.waitKey(1) == ord('q')):
        break;
    cv2.imshow("Color Image", color_image)
    cv2.waitKey(1)

out.release()
