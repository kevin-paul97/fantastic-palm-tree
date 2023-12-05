import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# Load the pretrained keypoint detection model
model = keypointrcnn_resnet50_fpn(pretrained=True, device="mps")
model.eval()

# Define transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Grayscale()])

# Set up the video capture
cap = cv2.VideoCapture(0)  # Change to the appropriate camera index if not the default

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Transform the frame
    input_tensor = transform(rgb_frame)
    input_batch = input_tensor.unsqueeze(0)
    input_batch.to("mps")


    # Make prediction
    with torch.no_grad():
        prediction = model(input_batch)

    # Get keypoints from the prediction
    keypoints = prediction[0]['keypoints'][0].numpy()

    # Draw keypoints on the frame
    for keypoint in keypoints:
        x, y, _ = keypoint.astype(int)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow('Keypoint Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
