import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

screen_width, screen_height = pyautogui.size()
scaling_factor = screen_width / 0.9

alpha = 0.7  # Smoothing factor (0 to 1, higher means smoother)
smoothed_landmarks = None

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
        index_tip = mp_hands.HandLandmark.INDEX_FINGER_TIP
        thumb_tip_x, thumb_tip_y = hand_landmarks.landmark[thumb_tip].x, hand_landmarks.landmark[thumb_tip].y
        index_tip_x, index_tip_y = hand_landmarks.landmark[index_tip].x, hand_landmarks.landmark[index_tip].y

        if smoothed_landmarks is None:
          smoothed_landmarks = hand_landmarks
        else:
          for i in range(len(hand_landmarks.landmark)):
              landmark = hand_landmarks.landmark[i]
              smoothed_landmark = smoothed_landmarks.landmark[i]
              smoothed_landmark.x = alpha * landmark.x + (1 - alpha) * smoothed_landmark.x
              smoothed_landmark.y = alpha * landmark.y + (1 - alpha) * smoothed_landmark.y

            # Use smoothed landmarks for further processing
        scaled_x = int(smoothed_landmarks.landmark[index_tip].x * scaling_factor)
        scaled_y = int(smoothed_landmarks.landmark[index_tip].y * scaling_factor)

      # Scale coordinates to screen size
        #scaled_x = int(index_tip_x * scaling_factor)
        #scaled_y = int(index_tip_y * scaling_factor)

      # Move mouse pointer
        try:
            pyautogui.moveTo(scaled_x, scaled_y)
        except pyautogui.exceptions.PyAutoGUIError:
            print("Error moving mouse pointer. Ignoring.")
        
        distance = ((index_tip_x - thumb_tip_x) ** 2 + (index_tip_y - thumb_tip_y) ** 2) ** 0.5
        if distance < 0.05:  # If fingers are close enough (approximate)
          pyautogui.click()
    cv2.imshow('Handtracking', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
