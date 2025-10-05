import cv2

# Try AVFoundation on macOS for better device access
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open Insta360 camera.")
    exit()

# Set expected 360° output resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_FPS, 60)

print("Press 'q' to quit.")
print("Expecting 2880x1440 frame size from Insta360 X5.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Can't receive frame (stream end?). Retrying...")
        continue

    # Show frame shape (width × height)
    print(f"Frame shape: {frame.shape}", end="\r")

    # Display the full equirectangular image
    cv2.imshow("Insta360 X5 - 360° Equirectangular View", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
