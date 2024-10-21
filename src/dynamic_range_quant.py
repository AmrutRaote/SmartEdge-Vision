#dynamic_range_quant

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import time

# Load the dynamic_range_quant model
model_path_dynamic_range_quant = '/home/pi/Desktop/OBJECT DETECTION/MODELS/best_dynamic_range_quant.tflite'
interpreter_dynamic_range_quant = tf.lite.Interpreter(model_path=model_path_dynamic_range_quant)
interpreter_dynamic_range_quant.allocate_tensors()

input_details_dynamic_range_quant = interpreter_dynamic_range_quant.get_input_details()
output_details_dynamic_range_quant = interpreter_dynamic_range_quant.get_output_details()

# Get height and width of image from input tensor
image_height = input_details_dynamic_range_quant[0]['shape'][1]
image_width = input_details_dynamic_range_quant[0]['shape'][2]

image_name = 'cars.jpg'
image = Image.open(f'/home/pi/Desktop/OBJECT DETECTION/{image_name}')
image_resized = image.resize((image_width, image_height))

# Preprocess image for dynamic_range_quant model
image_np_dynamic_range_quant = np.array(image_resized, dtype=np.float32)
image_np_dynamic_range_quant /= 255.0 
image_np_dynamic_range_quant = np.expand_dims(image_np_dynamic_range_quant, axis=0)

start_preprocess = time.time()

# Ensure that the input type matches the expected type
interpreter_dynamic_range_quant.set_tensor(input_details_dynamic_range_quant[0]['index'], image_np_dynamic_range_quant)

# Inference Time
start_inference_dynamic_range_quant = time.time()
interpreter_dynamic_range_quant.invoke()
inference_time_dynamic_range_quant = time.time() - start_inference_dynamic_range_quant

# output results for dynamic_range_quant
output_dynamic_range_quant = interpreter_dynamic_range_quant.get_tensor(output_details_dynamic_range_quant[0]['index'])
output_dynamic_range_quant = output_dynamic_range_quant[0]
output_dynamic_range_quant = output_dynamic_range_quant.T

# Postprocess Time
start_postprocess = time.time()

# Bounding box coordinates
boxes_xywh_dynamic_range_quant = output_dynamic_range_quant[..., :4]
scores_dynamic_range_quant = np.max(output_dynamic_range_quant[..., 5:], axis=1)
classes_dynamic_range_quant = np.argmax(output_dynamic_range_quant[..., 5:], axis=1)

# Threshold Setting
threshold = 0.25

# Bounding boxes
draw = ImageDraw.Draw(image_resized)

for box, score, cls in zip(boxes_xywh_dynamic_range_quant, scores_dynamic_range_quant, classes_dynamic_range_quant):
    if score >= threshold:
        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
        text = f"Class: {cls}, Score: {score:.2f}"
        draw.text((x1, y1), text)

image_resized.save(f"/home/pi/Desktop/OBJECT DETECTION/OUTPUT/dynamic_range_quant_detected_{image_name}")

print(f'Preprocess Time: {(start_inference_dynamic_range_quant - start_preprocess) * 1000:.2f}ms')
print(f'Inference Time (dynamic_range_quant): {(inference_time_dynamic_range_quant) * 1000:.2f}ms')
print(f'Postprocess Time: {(time.time() - start_postprocess) * 1000:.2f}ms')
