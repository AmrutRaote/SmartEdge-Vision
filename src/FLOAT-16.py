##FOR FLOAT-16

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import time

model_path_float16 = '/home/pi/Desktop/OBJECT DETECTION/MODELS/best_float16.tflite'
interpreter_float16 = tf.lite.Interpreter(model_path=model_path_float16)
interpreter_float16.allocate_tensors()

input_details_float16 = interpreter_float16.get_input_details()
output_details_float16 = interpreter_float16.get_output_details()

image_height = input_details_float16[0]['shape'][1]
image_width = input_details_float16[0]['shape'][2]

image_name = 'cars.jpg'
image = Image.open(f'/home/pi/Desktop/OBJECT DETECTION/{image_name}')
image_resized = image.resize((image_width, image_height))

# Preprocess image 
image_np_float16 = np.array(image_resized, dtype=np.float32)
image_np_float16 = np.true_divide(image_np_float16, 255)

# Ensure 4D input tensor by adding a batch dimension
image_np_float16 = np.expand_dims(image_np_float16, axis=0)

start_preprocess = time.time()

interpreter_float16.set_tensor(input_details_float16[0]['index'], image_np_float16)

# Inference Time
start_inference_float16 = time.time()
interpreter_float16.invoke()
inference_time_float16 = time.time() - start_inference_float16

# output results
output_float16 = interpreter_float16.get_tensor(output_details_float16[0]['index'])
output_float16 = output_float16[0]
output_float16 = output_float16.T
# Postprocess 
start_postprocess = time.time()

boxes_xywh_float16 = output_float16[..., :4]
scores_float16 = np.max(output_float16[..., 5:], axis=1)
classes_float16 = np.argmax(output_float16[..., 5:], axis=1)

# Threshold Setting
threshold = 0.25

# Bounding boxes
draw = ImageDraw.Draw(image_resized)

for box, score, cls in zip(boxes_xywh_float16, scores_float16, classes_float16):
    if score >= threshold:
        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
        text = f"Class: {cls}, Score: {score:.2f}"
        draw.text((x1, y1), text)

image_resized.save(f"/home/pi/Desktop/OBJECT DETECTION/OUTPUT/float16_detected_{image_name}")

print(f'Preprocess Time: {(start_inference_float16 - start_preprocess) * 1000:.2f}ms')
print(f'Inference Time (float16): {(inference_time_float16) * 1000:.2f}ms')
print(f'Postprocess Time: {(time.time() - start_postprocess) * 1000:.2f}ms')
