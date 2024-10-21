##FOR FLOAT-32
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import time

model_path = '/home/pi/Desktop/OBJECT DETECTION/MODELS/best_float32.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# get height and width of image from input tensor
image_height = input_details[0]['shape'][1]
image_width = input_details[0]['shape'][2]

# Image Preparation
image_name = 'cars1.jpg'
image = Image.open(f'/home/pi/Desktop/OBJECT DETECTION/{image_name}')
image_resized = image.resize((image_width, image_height)) # Resize the image for input tensor and store it in variable

image_np = np.array(image_resized) #
image_np = np.true_divide(image_np, 255, dtype=np.float32)
image_np = image_np[np.newaxis, :]

# Preprocess Time
start_preprocess = time.time()

# inference
interpreter.set_tensor(input_details[0]['index'], image_np)

# Inference Time
start_inference = time.time()
interpreter.invoke()
inference_time = time.time() - start_inference

# Obtaining output results
output = interpreter.get_tensor(output_details[0]['index'])
output = output[0]
output = output.T

# Postprocess Time
start_postprocess = time.time()

boxes_xywh = output[..., :4] #Get coordinates of bounding box, first 4 columns of output tensor
scores = np.max(output[..., 5:], axis=1) #Get score value, 5th column of output tensor
classes = np.argmax(output[..., 5:], axis=1) # Get the class value, get the 6th and subsequent columns of the output tensor, and store the largest value in the output tensor.

# Threshold Setting
threshold = 0.25

# Bounding boxes, scores, and classes are drawn on the image
draw = ImageDraw.Draw(image_resized)

for box, score, cls in zip(boxes_xywh, scores, classes):
    if score >= threshold:
        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
        text = f"Class: {cls}, Score: {score:.2f}"
        draw.text((x1, y1), text)

# Saving Images
image_resized.save(f"/home/pi/Desktop/OBJECT DETECTION/OUTPUT/detected_{image_name}")

# Print timings
print(f'Preprocess Time: {(start_inference - start_preprocess) * 1000:.2f}ms')
print(f'Inference Time: {(inference_time) * 1000:.2f}ms')
print(f'Postprocess Time: {(time.time() - start_postprocess) * 1000:.2f}ms')
