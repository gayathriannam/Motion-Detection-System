import cv2
import time

# Set up camera object
cap = cv2.VideoCapture(0)

# Define background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Define motion detection area
motion_detection_area = None

# Loop through frames
while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Check if frame was successfully read
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction to frame
    fg_mask = bg_subtractor.apply(gray)

    # Apply binary threshold to foreground mask
    thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)[1]

    # Find contours in thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours and filter out small ones
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue

        # Compute bounding box of contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show frame with bounding boxes
    cv2.imshow("Motion Detection System", frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Exit loop if 'q' is pressed
    if key == ord("q"):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
