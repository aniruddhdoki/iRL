import cv2
import torch
import torchvision.transforms as T

# Load your PyTorch model
# model = YourModel()
# model.eval()  # Set the model to evaluation mode
model = torch.hub.load("ultralytics/yolov5", "yolov5s")
model.eval()

# Define a transform to convert frames to tensor and normalize (adjust as needed)
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((640, 640)),  # Resize the image to match the model's expected input size
    T.ToTensor(),
    # Normalize using mean and std values from your model training (if applicable)
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_objects(frame, model):
    # Apply the transformations and add batch dimension
    input_tensor = transform(frame).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model.cuda()
    
    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
    
    return outputs

# RTMP Stream URL from your local RTMP server
stream_url = 'rtmp://127.0.0.1/live/HJEvtL3qp'

# OpenCV to capture stream
cap = cv2.VideoCapture(stream_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection
    detections = detect_objects(frame, model)

    # Example visualization (adjust according to your model's output format)
    for detection in detections[0]:  # Assuming the model returns a batch of detections
        # Each detection is (x1, y1, x2, y2, score, class)
        x1, y1, x2, y2, score, class_id = map(int, detection)
        label = f"Class {class_id} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
