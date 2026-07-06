import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageSequence
import time

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def load_image_frames(path):
    try:
        img = Image.open(path)
        frames = []
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert("RGBA")
            cv_img = np.array(frame)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)
            frames.append(cv_img)
        return frames
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

images = {
    'default': load_image_frames('cats/default.webp'),
    'chup': load_image_frames('cats/chup.gif'),
    'lick': load_image_frames('cats/lick.gif'),
    'rocked': load_image_frames('cats/rocked.gif')
}

def get_current_frame(frames, start_time, fps=10):
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    
    elapsed = time.time() - start_time
    frame_index = int(elapsed * fps) % len(frames)
    return frames[frame_index]

def is_shaka_gesture(hand_landmarks):
    index_folded = hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y
    middle_folded = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_folded = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_extended = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    thumb_extended = abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[17].x) > abs(hand_landmarks.landmark[3].x - hand_landmarks.landmark[17].x)
    
    return index_folded and middle_folded and ring_folded and pinky_extended and thumb_extended

def is_shh_gesture(hand_landmarks, face_landmarks):
    index_extended = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_folded = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_folded = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_folded = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    
    if index_extended and middle_folded and ring_folded and pinky_folded:
        if face_landmarks:
            index_tip = hand_landmarks.landmark[8]
            mouth_top = face_landmarks.landmark[13]
            mouth_bottom = face_landmarks.landmark[14]
            
            dist_to_mouth_y = abs(index_tip.y - (mouth_top.y + mouth_bottom.y)/2)
            dist_to_mouth_x = abs(index_tip.x - (mouth_top.x + mouth_bottom.x)/2)
            
            if dist_to_mouth_x < 0.2 and dist_to_mouth_y < 0.2:
                return True
        else:
             return True
    return False

def is_tongue_out(face_landmarks):
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    mouth_openness = bottom_lip.y - top_lip.y
    return mouth_openness > 0.05

def main():
    cap = cv2.VideoCapture(1)
    
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_faces=1)
    
    current_gesture = 'default'
    gesture_start_time = time.time()
    
    print("Press 'q' or 'ESC' to exit.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
            
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        hand_results = hands.process(rgb_image)
        face_results = face_mesh.process(rgb_image)
        
        detected_gesture = 'default'
        
        face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if is_shaka_gesture(hand_landmarks):
                detected_gesture = 'rocked'
            elif is_shh_gesture(hand_landmarks, face_landmarks):
                detected_gesture = 'chup'
                
        if detected_gesture == 'default' and face_landmarks:
            if is_tongue_out(face_landmarks):
                detected_gesture = 'lick'
                
        if detected_gesture != current_gesture:
            current_gesture = detected_gesture
            gesture_start_time = time.time()
            
        cat_img_bgra = get_current_frame(images[current_gesture], gesture_start_time)
        if cat_img_bgra is not None:
            cv2.imshow('Cat Reaction', cat_img_bgra)
            
        cv2.imshow('Webcam Feed', image)
        
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
