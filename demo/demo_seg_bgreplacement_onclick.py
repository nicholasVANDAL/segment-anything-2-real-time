import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio

# Use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # Turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time

sam2_checkpoint = "../checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"

predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

cap = cv2.VideoCapture(0)  # Use webcam as input
background_image = cv2.imread('images/groceries.jpg')  # Load your background image
if_init = False
running = False
points = np.array([], dtype=np.float32).reshape(0, 2)
labels = np.array([], dtype=np.int32)

def mouse_callback(event, x, y, flags, param):
    global points, labels

    if event == cv2.EVENT_LBUTTONDOWN:
        points = np.append(points, [[x, y]], axis=0)
        labels = np.append(labels, [1])
        print(f"Added point at ({x}, {y}) with label 1")
    elif event == cv2.EVENT_RBUTTONDOWN:
        points = np.append(points, [[x, y]], axis=0)
        labels = np.append(labels, [0])
        print(f"Added point at ({x}, {y}) with label 0")

def draw_crosses(frame, points, labels):
    for i, point in enumerate(points):
        color = (0, 255, 0) if labels[i] == 1 else (0, 0, 255)  # Green for 1, Red for 0
        center = tuple(point.astype(int))
        size = 5  # Size of the cross
        cv2.drawMarker(frame, center, color, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=2)

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouse_callback)

while True:
    if not running:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw crosses on the frame at the specified points
        draw_crosses(frame, points, labels)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("s"):
            running = True
            if_init = False  # Reset for segmentation
            print("Starting video feed...")
            continue

        if cv2.waitKey(1) & 0xFF == ord("r"):
            points = np.array([], dtype=np.float32).reshape(0, 2)
            labels = np.array([], dtype=np.int32)
            print("Reset segmentation and captured new frame.")
            continue

    else:
        ret, frame = cap.read()
        if not ret:
            break

        width, height = frame.shape[:2][::-1]

        if not if_init:
            predictor.load_first_frame(frame)
            if_init = True

            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                frame_idx=0,
                obj_id=2,
                points=points,
                labels=labels,
            )

        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)

            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            for i in range(0, len(out_obj_ids)):
                out_mask = (
                    (out_mask_logits[i] > 0.0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                    * 255
                )

                all_mask = cv2.bitwise_or(all_mask, out_mask)

            # Invert the mask
            inverted_mask = cv2.bitwise_not(all_mask)

            # Resize the background image to match the frame size
            background_resized = cv2.resize(background_image, (width, height))

            # Use the inverted mask to copy the background where the mask is 0
            background_segment = cv2.bitwise_and(
                background_resized, background_resized, mask=inverted_mask
            )

            # Use the original mask to copy the frame where the mask is 255
            foreground_segment = cv2.bitwise_and(frame, frame, mask=all_mask)

            # Combine the background and foreground segments
            combined_frame = cv2.add(background_segment, foreground_segment)

            cv2.imshow("frame", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord("r"):
            running = False
            print("Paused video feed.")
            continue

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
