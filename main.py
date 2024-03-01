import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands



# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Convert the BGR image to RGB before processing.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the RGB image with MediaPipe Hands 
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for display
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        thumb_tip = mp_hands.HandLandmark.THUMB_TIP
        wrist = mp_hands.HandLandmark.WRIST
        thumb_tip_x, thumb_tip_y = hand_landmarks.landmark[thumb_tip].x, hand_landmarks.landmark[thumb_tip].y
        wrist_x, wrist_y = hand_landmarks.landmark[wrist].x, hand_landmarks.landmark[wrist].y

        # Check if thumb tip is above the wrist with a reasonable margin
        if thumb_tip_y < wrist_y - 0.5:  # Adjust threshold as needed
          print("Thumbs up detected!")
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
