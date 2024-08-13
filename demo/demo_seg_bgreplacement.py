import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time


sam2_checkpoint = "../checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

cap = cv2.VideoCapture(0)  # Use webcam as input
background_image = cv2.imread('images/cars.jpg')  # Load your background image
if_init = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    width, height = frame.shape[:2][::-1]

    if not if_init:
        predictor.load_first_frame(frame)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
        points = np.array([[240, 320]], dtype=np.float32)
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255

            all_mask = cv2.bitwise_or(all_mask, out_mask)


        # Invert the mask so that the object area is white (255) and the background is black (0)
        inverted_mask = cv2.bitwise_not(all_mask)

        # Resize the background image to match the frame size
        background_resized = cv2.resize(background_image, (width, height))

        # Use the inverted mask to copy the background where the mask is 0
        background_segment = cv2.bitwise_and(background_resized, background_resized, mask=inverted_mask)

        # Use the original mask to copy the frame where the mask is 255
        foreground_segment = cv2.bitwise_and(frame, frame, mask=all_mask)

        # Combine the background and foreground segments
        combined_frame = cv2.add(background_segment, foreground_segment)

        cv2.imshow("frame", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
