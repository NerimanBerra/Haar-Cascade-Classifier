import cv2
# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Start the video stream (use 0 for the default camera, or provide the video file path)
video_stream = cv2.VideoCapture(0)
while True:
    # Read a frame from the video stream
    ret, frame = video_stream.read()

    # Convert the frame to grayscale (required for face detection)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Blur the detected faces and keep the rest of the frame as it is
    for (x, y, w, h) in faces:
        # Get the region of interest (ROI) where the face is located
        face_roi = frame[y:y + h, x:x + w]

        # Apply Gaussian blur to the face ROI
        blurred_face = cv2.GaussianBlur(face_roi, (23, 23), 30)

        # Replace the face in the original frame with the blurred face
        frame[y:y + h, x:x + w] = blurred_face

    # Display the frame
    cv2.imshow('Live Video Stream with Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
video_stream.release()
cv2.destroyAllWindows()
