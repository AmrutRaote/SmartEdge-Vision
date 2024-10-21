import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import time
import cv2

model_path = '/home/pi/Desktop/OBJECT DETECTION/MODELS/best_float32.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get height and width of the input tensor
image_height = input_details[0]['shape'][1]
image_width = input_details[0]['shape'][2]

# Open the video file
video_path = '/home/pi/Desktop/OBJECT DETECTION/4K Video of Highway Traffic.mp4'
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('/home/pi/Desktop/OBJECT DETECTION/OUTPUT/f32_output_video.avi', fourcc, 20.0, (image_width, image_height))

# Initialize variables for timing
total_frame_count = 0
total_preprocess_time = 0
total_inference_time = 0
total_postprocess_time = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for input tensor
    frame_resized = cv2.resize(frame, (image_width, image_height))

    # Preprocess Time
    start_preprocess = time.time()

    # Normalize pixel values to [0, 1]
    frame_np = np.true_divide(frame_resized, 255, dtype=np.float32)
    frame_np = frame_np[np.newaxis, :]

    # Inference
    interpreter.set_tensor(input_details[0]['index'], frame_np)
    start_inference = time.time()
    interpreter.invoke()
    inference_time = time.time() - start_inference

    # Obtaining output results
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output[0]
    output = output.T

    # Postprocess Time
    start_postprocess = time.time()

    boxes_xywh = output[..., :4]  # Get coordinates of bounding box
    scores = np.max(output[..., 5:], axis=1)  # Get score value
    classes = np.argmax(output[..., 5:], axis=1)  # Get the class value

    # Threshold Setting
    threshold = 0.25

    # Draw bounding boxes, scores, and classes on the frame
    for box, score, cls in zip(boxes_xywh, scores, classes):
        if score >= threshold:
            x_center, y_center, width, height = box
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)

            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 1)
            text = f"Class: {cls}, Score: {score:.2f}"
            cv2.putText(frame_resized, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Saving Images
    output_video.write(frame_resized)

    # Update timing statistics
    total_preprocess_time += (start_inference - start_preprocess)
    total_inference_time += inference_time
    total_postprocess_time += (time.time() - start_postprocess)
    total_frame_count += 1

    # Print timings for each frame
    print(f'Frame {total_frame_count}:')
    print(f'  Preprocess Time: {(start_inference - start_preprocess) * 1000:.2f}ms')
    print(f'  Inference Time: {inference_time:.2f}ms')
    print(f'  Postprocess Time: {(time.time() - start_postprocess) * 1000:.2f}ms')

# Release video capture and writer
cap.release()
output_video.release()
cv2.destroyAllWindows()

# Print average timing statistics
print(f'Average Preprocess Time per Frame: {(total_preprocess_time / total_frame_count) * 1000:.2f}ms')
print(f'Average Inference Time per Frame: {(total_inference_time / total_frame_count) * 1000:.2f}ms')
print(f'Average Postprocess Time per Frame: {(total_postprocess_time / total_frame_count) * 1000:.2f}ms')
print(f'Total Time for {total_frame_count} Frames: {total_preprocess_time + total_inference_time + total_postprocess_time:.2f}s')

#!pip install ffmpeg-python
import ffmpeg
input_file_path = "/home/pi/Desktop/OBJECT DETECTION/OUTPUT/f32_output_video.avi"
output_file_path = "/home/pi/Desktop/OBJECT DETECTION/OUTPUT/f32_output_video.mp4"
input_video = ffmpeg.input(input_file_path)
output_video = input_video.output(output_file_path, format="mp4")
output_video.run()

from IPython.display import HTML
from base64 import b64encode
mp4 = open('/home/pi/Desktop/OBJECT DETECTION/OUTPUT/f32_output_video.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=512 height=288 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)



