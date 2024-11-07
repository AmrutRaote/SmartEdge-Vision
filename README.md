

# SmartEdge-Vision 🚀

The project focuses on implementing the YOLOv8n model for object detection on edge devices, particularly the Raspberry Pi. The project leverages model quantization techniques to optimize for efficiency while maintaining accuracy.

## Table of Contents 📚

- Project Overview
- Features
- Technologies Used
- Installation
  - Installation on Raspberry Pi
  - Installation on Google Colab
- Results
  - Performance Measurements
  - File Size Comparison
  - Processing Power Comparison
- Output
- Contributing

## Project Overview 💡

Object detection is a computer vision technique for identifying and locating objects in images or videos. This project focuses on developing a deep learning-based object detection model optimized for **edge devices**, such as the Raspberry Pi. We use the **YOLOv8n model**, a lightweight and efficient variant of the YOLO (You Only Look Once) architecture, and employ **model quantization** to reduce inference time and memory usage without significantly sacrificing accuracy.

### Performance Metrics 📊

The trained model was able to achieve the following results:
- Precision: 0.664
- Recall: 0.611
- mAP50: 0.668
- mAP50-95: 0.446

## Features ⭐

- **Real-Time Object Detection**: Utilizes the YOLOv8n model for fast and accurate object detection on edge devices.
- **Model Quantization**: Reduces model size and inference time by quantizing the model to lower precision formats (e.g., TensorFlow Lite, ONNX).
- **Edge Device Optimization**: The model is optimized to run efficiently on low-resource devices like the Raspberry Pi.


## Technologies Used ⚙️
- **YOLOv8n**: A lightweight version of YOLOv8 for real-time object detection.
- **TensorFlow Lite**: For efficient deployment on edge devices.
- **Google Colab**: Used for model training and initial performance testing.
- **Raspberry Pi**: Used as the target edge device for deployment.
- **Dataset**: [Link to Dataset](https://drive.google.com/file/d/17I2slDwb-4ns08Nsbs7FvS9ACQxg4ln4/view)
- **Models**: [Link to Models](https://drive.google.com/drive/folders/1-3RjEvFPCc1UquruDNMs0XgXrgt7l0rk)

## Installation 🔧

### Installation on Raspberry Pi

1. **Clone the project repository**:
   ```bash
   git clone https://github.com/AmrutRaote/SmartEdge-Vision
   ```

2. **Create a virtual environment**:
   Install `virtualenv` if you don't have it:
   ```bash
   sudo apt-get install python3-venv
   ```

   Then, create and activate the virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages**:
   The necessary dependencies are listed in the `/src/requirements.txt` file. Run the following command to install them:
   ```bash
   pip install -r src/requirements.txt
   ```

4. **Run the YOLOv8 model**:
   Use the following command to perform object detection, where `MODEL-PATH` is the path to the YOLO model you want to use (e.g., `models/yolov8n_integer_quant.tflite`), and `INPUT-IMAGE-VIDEO` is the path to the input image or video file.
   ```bash
   !yolo task=detect mode=predict model="MODEL-PATH" source="INPUT-IMAGE-VIDEO" imgsz=512
   ```

5. **Manual Model Execution**:
   The model can also be executed manually by navigating to the `/src/` folder and running the appropriate Python scripts.

---

### Installation on Google Colab

1. **Open the Google Colab notebook**:
   There is a notebook available in the `/src/GOOGLE_COLAB.ipynb` file, which contains all the necessary instructions for running the model on Google Colab.

## Results 🏆

### Performance Measurements
- **P**: Precision
- **R**: Recall
- **mAP50**: Mean Average Precision at IoU threshold 0.50
- **mAP50-95**: Mean Average Precision at IoU thresholds between 0.50 and 0.95

| Class          | Images | Instances | P     | R     | mAP50 | mAP50-95 |
|----------------|--------|-----------|-------|-------|-------|----------|
| **All**        | 8000   | 88824     | 0.664 | 0.611 | 0.668 | 0.446    |
| **Bus**        | 8000   | 4658      | 0.575 | 0.571 | 0.597 | 0.464    |
| **Car**        | 8000   | 60857     | 0.789 | 0.777 | 0.846 | 0.593    |
| **Person**     | 8000   | 7117      | 0.718 | 0.505 | 0.607 | 0.351    |
| **Traffic Light** | 8000 | 9592      | 0.704 | 0.548 | 0.647 | 0.308    |
| **Truck**      | 8000   | 6600      | 0.535 | 0.656 | 0.640 | 0.515    |





### File Size Comparison 📏
This table shows the file size comparison between the original YOLOv8n model and various quantized/compressed versions. The goal is to reduce file size and optimize model performance for edge devices without significantly compromising accuracy.

| Model                                 | File Size (MB) |
|---------------------------------------|----------------|
| YOLOv8n.pt                            | 5.98 MB        |
| YOLOv8n_float32.tflite                | 11.65 MB       |
| YOLOv8n_float16.tflite                | 5.87 MB        |
| YOLOv8n_integer_quant.tflite           | 2.98 MB        |
| YOLOv8n_int8.tflite                   | 3.07 MB        |
| YOLOv8n_dynamic_range_quant.tflite     | 3.06 MB        |
| YOLOv8n_integer_quant_with_int16_act.tflite | 3.04 MB        |
| YOLOv8n_full_integer_quant.tflite      | 3.00 MB        |
| YOLOv8n.onnx (ONNX Runtime)           | 11.60 MB       |

![image](https://github.com/user-attachments/assets/b16f6313-afa0-4a5f-8fcb-a85082685f44)


## Processing Power Comparison ⚡
These tables provide a detailed comparison of the different YOLOv8 models in terms of processing times across two platforms: **Google Colab (T4 GPU)**, which represents a high-performance environment, and **Raspberry Pi 4**, a low-power edge device.

### Google Colab - High Processing Power (T4 GPU)

| Model Name                                      | Preprocessing Time (ms) | Inference Time (ms) | Postprocessing Time (ms) |
|-------------------------------------------------|-------------------------|---------------------|--------------------------|
| **YOLOv8n.pt**                                  | 4.6                     | 41.9                | 1206.1                   |
| **YOLOv8n_float32.tflite**                      | 1.86                    | 121.51              | 16.4                     |
| **YOLOv8n_float16.tflite**                      | 0.9                     | 120.13              | 16.28                    |
| **YOLOv8n_integer_quant.tflite**                | 1.91                    | 97.21               | 15.64                    |
| **YOLOv8n_int8.tflite**                         | 2.17                    | 174.18              | 15.5                     |
| **YOLOv8n_dynamic_range_quant.tflite**          | 2.96                    | 257.48              | 31.7                     |
| **YOLOv8n_integer_quant_with_int16_act.tflite** | 0.68                    | 1658.25             | 17.17                    |
| **YOLOv8n_full_integer_quant.tflite**           | 3.9                     | 248.4               | 2.5                      |
| **YOLOv8n.onnx (ONNX Runtime)**                 | 28.3                    | 122.8               | 549.8                    |

### Raspberry Pi 4 Model B (8GB RAM) - Low Processing Power

| Model Name                                      | Preprocessing Time (ms) | Inference Time (ms) | Postprocessing Time (ms) |
|-------------------------------------------------|-------------------------|---------------------|--------------------------|
| **YOLOv8n.pt**                                  | 20.9                    | 1534.2              | 52.4                     |
| **YOLOv8n_float32.tflite**                      | 3.86                    | 669.22              | 243.45                   |
| **YOLOv8n_float16.tflite**                      | 3.62                    | 639.16              | 244.77                   |
| **YOLOv8n_integer_quant.tflite**                | 4.47                    | 356.91              | 255.34                   |
| **YOLOv8n_int8.tflite**                         | 5.52                    | 630.79              | 241.1                    |
| **YOLOv8n_dynamic_range_quant.tflite**          | 5.76                    | 685.09              | 251.6                    |
| **YOLOv8n_integer_quant_with_int16_act.tflite** | 6.33                    | 2864.67             | 240.25                   |
| **YOLOv8n_full_integer_quant.tflite**           | 10.9                    | 478.6               | 4.3                      |
| **YOLOv8n.onnx (ONNX Runtime)**                 | 112                     | 460.2               | 149.8                    |

![image](https://github.com/user-attachments/assets/8756b17e-4de2-4118-bf7f-3796c24408e5)

## Output 📸

  ![cars](https://github.com/user-attachments/assets/1adb5c48-1140-461d-9f63-a6f1401ac40f)



  https://github.com/user-attachments/assets/4e56e9d8-611e-45f0-a056-f33f72cd317a


  https://github.com/user-attachments/assets/9ed7049a-2268-4abe-ade4-76b2f1bbfd9d



## Contributing 🤝

Contributions are welcome! Feel free to open issues or submit pull requests to help improve the project.
