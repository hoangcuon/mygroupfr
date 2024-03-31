import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
model_path = 'C:\\Users\\zincuonn\\Documents\\Python\\model_test\\tflite\\tflite_16\\1.tflite'  # Update with the path to your TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Load class labels
classes = ['Fire', 'Fire-', 'Smoke']

# Define confidence threshold
conf_threshold = 0.01

# Initialize video capture from default camera (index 0)
cap = cv2.VideoCapture(0)

# Open file for saving detection results
output_file = open('detection_results.txt', 'w')

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32)

    # Set model input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Perform inference
    interpreter.invoke()

    # Retrieve output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Postprocess detections
    for detection in output_data[0]:
        score = float(detection[2])
        if score > conf_threshold:
            class_id = int(detection[1])
            label = classes[class_id]
            confidence = score
            output_file.write(f'{label}: {confidence:.2f}\n')

    # Display the frame
    cv2.imshow('Detection Results', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close the output file
output_file.close()
