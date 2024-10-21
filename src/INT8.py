#INT 8
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import time

# Load the int8 model
model_path_int8 = '/home/pi/Desktop/OBJECT DETECTION/MODELS/best_int8.tflite'
interpreter_int8 = tf.lite.Interpreter(model_path=model_path_int8)
interpreter_int8.allocate_tensors()

input_details_int8 = interpreter_int8.get_input_details()
output_details_int8 = interpreter_int8.get_output_details()

# Get height and width of image from input tensor
image_height = input_details_int8[0]['shape'][1]
image_width = input_details_int8[0]['shape'][2]

# Image Preparation
image_name = 'cars.jpg'
image = Image.open(f'/home/pi/Desktop/OBJECT DETECTION/{image_name}')
image_resized = image.resize((image_width, image_height))

# Preprocess the image for int8 model
image_np_int8 = np.array(image_resized, dtype=np.float32)  # Change dtype to np.float32
image_np_int8 = np.true_divide(image_np_int8, 255.0)  # Normalize the pixel values to the range [0, 1]
image_np_int8 = np.expand_dims(image_np_int8, axis=0)

# Measure Preprocess Time
start_preprocess = time.time()

# Ensure that the input type matches the expected type
interpreter_int8.set_tensor(input_details_int8[0]['index'], image_np_int8)

# Inference Time
start_inference_int8 = time.time()
interpreter_int8.invoke()
inference_time_int8 = time.time() - start_inference_int8

# Obtaining output results for int8
output_int8 = interpreter_int8.get_tensor(output_details_int8[0]['index'])
output_int8 = output_int8[0]
output_int8 = output_int8.T

# Postprocess Time
start_postprocess = time.time()

# Bounding box coordinates, scores, and classes
boxes_xywh_int8 = output_int8[..., :4]
scores_int8 = np.max(output_int8[..., 5:], axis=1)
classes_int8 = np.argmax(output_int8[..., 5:], axis=1)

# Threshold Setting
threshold = 0.25

# Bounding boxes, scores, and classes are drawn on the image
draw = ImageDraw.Draw(image_resized)

for box, score, cls in zip(boxes_xywh_int8, scores_int8, classes_int8):
    if score >= threshold:
        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
        text = f"Class: {cls}, Score: {score:.2f}"
        draw.text((x1, y1), text)

# Save the image
image_resized.save(f"/home/pi/Desktop/OBJECT DETECTION/OUTPUT/int8_detected_{image_name}")

# Print timings
print(f'Preprocess Time: {(start_inference_int8 - start_preprocess) * 1000:.2f}ms')
print(f'Inference Time (int8): {(inference_time_int8) * 1000:.2f}ms')
print(f'Postprocess Time: {(time.time() - start_postprocess) * 1000:.2f}ms')
