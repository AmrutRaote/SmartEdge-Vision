#best_integer_quant_with_int16_act.tflite

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import time

# Load the integer_quant_with_int16_act model
model_path_integer_quant_with_int16_act = '/home/pi/Desktop/OBJECT DETECTION/MODELS/best_integer_quant_with_int16_act.tflite'
interpreter_integer_quant_with_int16_act = tf.lite.Interpreter(model_path=model_path_integer_quant_with_int16_act)
interpreter_integer_quant_with_int16_act.allocate_tensors()

input_details_integer_quant_with_int16_act = interpreter_integer_quant_with_int16_act.get_input_details()
output_details_integer_quant_with_int16_act = interpreter_integer_quant_with_int16_act.get_output_details()

# Get height and width of image from input tensor
image_height = input_details_integer_quant_with_int16_act[0]['shape'][1]
image_width = input_details_integer_quant_with_int16_act[0]['shape'][2]

# Image Preparation
image_name = 'cars.jpg'
image = Image.open(f'/home/pi/Desktop/OBJECT DETECTION/{image_name}')
image_resized = image.resize((image_width, image_height))

# Preprocess the image for integer_quant_with_int16_act model
image_np_integer_quant_with_int16_act = np.array(image_resized, dtype=np.float32)  # Change dtype to np.float32
image_np_integer_quant_with_int16_act /= 255.0  # Normalize the pixel values to the range [0, 1]
image_np_integer_quant_with_int16_act = np.expand_dims(image_np_integer_quant_with_int16_act, axis=0)

# Measure Preprocess Time
start_preprocess = time.time()

# Ensure that the input type matches the expected type
interpreter_integer_quant_with_int16_act.set_tensor(
    input_details_integer_quant_with_int16_act[0]['index'], image_np_integer_quant_with_int16_act
)

# Inference Time
start_inference_integer_quant_with_int16_act = time.time()
interpreter_integer_quant_with_int16_act.invoke()
inference_time_integer_quant_with_int16_act = time.time() - start_inference_integer_quant_with_int16_act

# Obtaining output results for integer_quant_with_int16_act
output_integer_quant_with_int16_act = interpreter_integer_quant_with_int16_act.get_tensor(
    output_details_integer_quant_with_int16_act[0]['index']
)
output_integer_quant_with_int16_act = output_integer_quant_with_int16_act[0]
output_integer_quant_with_int16_act = output_integer_quant_with_int16_act.T

# Postprocess Time
start_postprocess = time.time()

# Bounding box coordinates, scores, and classes
boxes_xywh_integer_quant_with_int16_act = output_integer_quant_with_int16_act[..., :4]
scores_integer_quant_with_int16_act = np.max(output_integer_quant_with_int16_act[..., 5:], axis=1)
classes_integer_quant_with_int16_act = np.argmax(output_integer_quant_with_int16_act[..., 5:], axis=1)

# Threshold Setting
threshold = 0.25

# Bounding boxes, scores, and classes are drawn on the image
draw = ImageDraw.Draw(image_resized)

for box, score, cls in zip(
    boxes_xywh_integer_quant_with_int16_act, scores_integer_quant_with_int16_act, classes_integer_quant_with_int16_act
):
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
image_resized.save(f"/home/pi/Desktop/OBJECT DETECTION/OUTPUT/integer_quant_with_int16_act_detected_{image_name}")

# Print timings
print(f'Preprocess Time: {(start_inference_integer_quant_with_int16_act - start_preprocess) * 1000:.2f}ms')
print(
    f'Inference Time (integer_quant_with_int16_act): {(inference_time_integer_quant_with_int16_act) * 1000:.2f}ms')
print(f'Postprocess Time: {(time.time() - start_postprocess) * 1000:.2f}ms')
