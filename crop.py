import cv2
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import numpy as np

current_roi = [0, 0, 0, 0]


def choose_video_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Choose video to crop",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
    )
    return file_path if file_path else None


def draw_roi_with_text(img, roi):
    x, y, w, h = roi
    color = (0, 255, 0) if w > 0 and h > 0 else (0, 0, 255)

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    text = f"{x} {y} {w} {h}"
    cv2.putText(img, text,
                (x, y - 10 if y > 20 else y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(img,
                "Drag mouse to choose - Enter: Confirm - Esc: Cancel",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA)


def mouse_callback(event, x, y, flags, param):
    global current_roi

    frame, window_name = param
    img = frame.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        current_roi[0] = x
        current_roi[1] = y
        current_roi[2] = 0
        current_roi[3] = 0

    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
        current_roi[2] = x - current_roi[0]
        current_roi[3] = y - current_roi[1]
        draw_roi_with_text(img, current_roi)
        cv2.imshow(window_name, img)

    elif event == cv2.EVENT_LBUTTONUP:
        current_roi[2] = x - current_roi[0]
        current_roi[3] = y - current_roi[1]

        if current_roi[2] < 0:
            current_roi[0] += current_roi[2]
            current_roi[2] = abs(current_roi[2])
        if current_roi[3] < 0:
            current_roi[1] += current_roi[3]
            current_roi[3] = abs(current_roi[3])

        draw_roi_with_text(img, current_roi)
        cv2.imshow(window_name, img)


def select_roi_with_realtime_coords(first_frame):
    global current_roi
    current_roi = [0, 0, 0, 0]

    window_name = "Select crop area (real-time coordinates)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, (first_frame, window_name))

    img = first_frame.copy()
    draw_roi_with_text(img, current_roi)
    cv2.imshow(window_name, img)

    print("\n=== INSTRUCTIONS ===")
    print("• Drag left mouse button to select crop region")
    print("• Enter or Space: Confirm")
    print("• Esc: Cancel (crop full video)")

    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key in (13, 32):  
        x, y, w, h = current_roi
        
        x = max(0, x)
        y = max(0, y)
        w = max(10, min(w, first_frame.shape[1] - x))
        h = max(10, min(h, first_frame.shape[0] - y))

        if w >= 10 and h >= 10:
            return (x, y, w, h)
        else:
            print("Selected area too small (< 10x10) → canceled")
            return None
    else:
        return None


def ask_folder_name(default_name):
    root = tk.Tk()
    root.withdraw()

    folder_name = simpledialog.askstring(
        title="Name the output folder",
        prompt="Enter the name for the output folder (default is video name):",
        initialvalue=default_name
    )

    return folder_name.strip() if folder_name and folder_name.strip() else default_name


def crop_video(input_video_path, roi):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Cannot open video!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    folder_name = ask_folder_name(video_name)
    output_folder = folder_name
    os.makedirs(output_folder, exist_ok=True)

    output_video_path = os.path.join(output_folder, f"{video_name}_crop.mp4")
    roi_txt_path = os.path.join(output_folder, "roi.txt")

    print(f"\nOutput folder:       {output_folder}")
    print(f"Output video:        {output_video_path}")

    if roi is None:
        out_size = (frame_width, frame_height)
        print("Cropping full video (no crop).")
    else:
        x, y, w, h = roi
        out_size = (w, h)
        print(f"Crop region: x={x}, y={y}, w={w}, h={h}")

        with open(roi_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{x} {y} {w} {h}\n")
        print(f"Saved roi.txt to: {roi_txt_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, out_size)

    if not out.isOpened():
        print("Error: Cannot create output video!")
        cap.release()
        return

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing... (total {total_frames} frames)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if roi:
            cropped = frame[y:y+h, x:x+w]
        else:
            cropped = frame

        out.write(cropped)
        frame_count += 1

        if frame_count % 500 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"\nDONE! Video saved at:\n{output_video_path}")


if __name__ == "__main__":
    print("=== VIDEO CROP TOOL - Custom folder + roi.txt ===\n")

    video_path = choose_video_file()
    if not video_path:
        print("No video selected → exit.")
        sys.exit(0)

    print(f"Input video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("Cannot read first frame!")
        sys.exit(1)

    selected_roi = select_roi_with_realtime_coords(first_frame)

    crop_video(video_path, selected_roi)