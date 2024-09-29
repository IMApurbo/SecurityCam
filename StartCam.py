import cv2
import mediapipe as mp
import time
import os
import telebot

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Telegram Bot with your bot token.
bot_token = 'YOUR_TELEGRAM_BOT_TOKEN'
chat_id = 'USER_ID_OF_THE_USER'  # Replace with the chat ID to which you want to send the image
bot = telebot.TeleBot(bot_token)

# Directory to save images.
image_dir = "detected_images"
os.makedirs(image_dir, exist_ok=True)

# Open the webcam.
cap = cv2.VideoCapture(0)

# Flags and variables for tracking
person_in_frame = False
photo_taken = False
start_time = None

# Function to save the frame as an image and send it via Telegram.
def save_and_send_photo(image):
    timestamp = int(time.time())
    filename = f"person_detected_{timestamp}.jpg"
    filepath = os.path.join(image_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"Photo saved: {filepath}")
    
    # Send the photo via Telegram bot
    with open(filepath, 'rb') as photo:
        bot.send_photo(chat_id, photo)
    print(f"Photo sent via Telegram to chat ID {chat_id}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
    # Process the image to find poses.
    results = pose.process(image)
    
    # Convert the image back to BGR for OpenCV.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Check if any pose landmarks were detected.
    if results.pose_landmarks:
        # Get the bounding box around the person.
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        for lm in results.pose_landmarks.landmark:
            x_min = min(x_min, lm.x)
            y_min = min(y_min, lm.y)
            x_max = max(x_max, lm.x)
            y_max = max(y_max, lm.y)

        h, w, _ = image.shape
        x_min, y_min = int(x_min * w), int(y_min * h)
        x_max, y_max = int(x_max * w), int(y_max * h)

        # Draw a green bounding box around the person.
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Check if the person is fully in the frame (e.g., face is visible).
        if not person_in_frame:
            # Check the visibility of key landmarks like the nose or eyes to ensure the face is fully in the frame.
            face_landmarks = [
                mp_pose.PoseLandmark.NOSE, 
                mp_pose.PoseLandmark.LEFT_EYE_INNER, 
                mp_pose.PoseLandmark.RIGHT_EYE_INNER
            ]
            if all(results.pose_landmarks.landmark[lm].visibility > 0.5 for lm in face_landmarks):
                # Start the timer when the person is first detected with a visible face.
                start_time = time.time()
                person_in_frame = True
                photo_taken = False
        
        # Check if 3 seconds have passed since the person entered the frame.
        elif person_in_frame and not photo_taken:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 3:
                save_and_send_photo(frame)
                photo_taken = True

    else:
        # If no person is detected in the frame, reset the flags and timer.
        if person_in_frame:
            print("Person exited the frame before 3 seconds.")
        person_in_frame = False
        photo_taken = False
        start_time = None

    # Display the image with the bounding box.
    cv2.imshow('MediaPipe Pose with Bounding Box', image)

    # Break the loop if 'q' is pressed.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the webcam and close windows.
cap.release()
cv2.destroyAllWindows()
