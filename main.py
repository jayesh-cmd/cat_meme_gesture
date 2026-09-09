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
    'rocked': load_image_frames('cats/rocked.gif'),
    'goback': load_image_frames('cats/goback.jpg'),
    'join_hand': load_image_frames('cats/join_hand.jpg'),
    'like': load_image_frames('cats/like.jpg'),
    'stare': load_image_frames('cats/stare.jpg'),
    'thinking': load_image_frames('cats/thinking.png'),
    'tongue_out': load_image_frames('cats/toung out.jpg')
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

def is_single_hand_prayer(hand_landmarks):
    index_extended = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_extended = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    ring_extended = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
    pinky_extended = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    
    if not (index_extended and middle_extended and ring_extended and pinky_extended):
        return False
        
    wrist = hand_landmarks.landmark[0]
    middle_tip = hand_landmarks.landmark[12]
    
    dx = abs(wrist.x - middle_tip.x)
    dy = abs(wrist.y - middle_tip.y)
    is_vertical = dy > 1.5 * dx
    
    thumb_tip = hand_landmarks.landmark[4]
    index_mcp = hand_landmarks.landmark[5]
    dist_thumb_index = np.sqrt((thumb_tip.x - index_mcp.x)**2 + (thumb_tip.y - index_mcp.y)**2)
    
    return is_vertical and dist_thumb_index < 0.1

def check_index_finger_gesture(hand_landmarks, face_landmarks, handedness):
    if not hand_landmarks or not face_landmarks:
        return 'default'
        
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    
    index_extended = index_tip.y < index_pip.y
    middle_folded = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_folded = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_folded = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    
    if not (index_extended and middle_folded and ring_folded and pinky_folded):
        return 'default'
        
    chin = face_landmarks.landmark[152]
    left_cheek = face_landmarks.landmark[234]
    right_cheek = face_landmarks.landmark[454]
    mouth_top = face_landmarks.landmark[13]
    mouth_bottom = face_landmarks.landmark[14]
    
    dist_chin = np.sqrt((index_tip.x - chin.x)**2 + (index_tip.y - chin.y)**2)
    dist_l_cheek = np.sqrt((index_tip.x - left_cheek.x)**2 + (index_tip.y - left_cheek.y)**2)
    dist_r_cheek = np.sqrt((index_tip.x - right_cheek.x)**2 + (index_tip.y - right_cheek.y)**2)
    dist_mouth = np.sqrt((index_tip.x - (mouth_top.x + mouth_bottom.x)/2)**2 + (index_tip.y - (mouth_top.y + mouth_bottom.y)/2)**2)
    
    near_face = min(dist_chin, dist_l_cheek, dist_r_cheek, dist_mouth) < 0.2
    
    if near_face:
        label = handedness.classification[0].label
        if label == 'Left':
            return 'chup'
        else:
            return 'thinking'
            
    return 'default'

def is_tongue_out(face_landmarks):
    top_lip = face_landmarks.landmark[13]
    bottom_lip = face_landmarks.landmark[14]
    mouth_openness = bottom_lip.y - top_lip.y
    return mouth_openness > 0.05

def is_thumbs_up(hand_landmarks):
    thumb_up = hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y and hand_landmarks.landmark[4].y < hand_landmarks.landmark[5].y
    index_folded = hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y
    middle_folded = hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y
    ring_folded = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
    pinky_folded = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
    return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded

def are_hands_joined(hand_results):
    if not hand_results.multi_hand_landmarks or len(hand_results.multi_hand_landmarks) < 2:
        return False
        
    hand1 = hand_results.multi_hand_landmarks[0]
    hand2 = hand_results.multi_hand_landmarks[1]
    
    center1_x = sum([lm.x for lm in hand1.landmark]) / 21.0
    center1_y = sum([lm.y for lm in hand1.landmark]) / 21.0
    center2_x = sum([lm.x for lm in hand2.landmark]) / 21.0
    center2_y = sum([lm.y for lm in hand2.landmark]) / 21.0
    
    dist = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    # A generous threshold to detect if two hands are generally close to each other
    return dist < 0.3

def is_back_hand(hand_landmarks, handedness):
    index_extended = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_extended = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    ring_extended = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
    pinky_extended = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
    
    if not (index_extended and middle_extended and ring_extended and pinky_extended):
        return False
        
    thumb_tip = hand_landmarks.landmark[4]
    index_mcp = hand_landmarks.landmark[5]
    dist_thumb_index = np.sqrt((thumb_tip.x - index_mcp.x)**2 + (thumb_tip.y - index_mcp.y)**2)
    
    # "Go back" / Stop usually has the thumb extended out, not tucked closely.
    return dist_thumb_index >= 0.1

def is_stare(face_landmarks):
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    
    angle = 0
    if dx != 0:
        angle = np.abs(np.degrees(np.arctan2(dy, dx)))
        
    nose = face_landmarks.landmark[1]
    eyes_center = face_landmarks.landmark[168]
    chin = face_landmarks.landmark[152]
    
    dist_eyes_nose = np.sqrt((nose.x - eyes_center.x)**2 + (nose.y - eyes_center.y)**2)
    dist_nose_chin = np.sqrt((nose.x - chin.x)**2 + (nose.y - chin.y)**2)
    
    pitch_ratio = dist_nose_chin / (dist_eyes_nose + 1e-6)
    
    # Detect if face is tilted sideways (>10 deg) OR if face is pitched down (Kubrick stare, ratio < 0.95)
    return (angle > 10 and angle < 170) or pitch_ratio < 0.95



def main():
    cap = cv2.VideoCapture(1)
    
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
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
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if are_hands_joined(hand_results):
                detected_gesture = 'join_hand'
            else:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                handedness = hand_results.multi_handedness[0]
                
                idx_gesture = check_index_finger_gesture(hand_landmarks, face_landmarks, handedness)
                
                if is_single_hand_prayer(hand_landmarks):
                    detected_gesture = 'join_hand'
                elif idx_gesture != 'default':
                    detected_gesture = idx_gesture
                elif is_shaka_gesture(hand_landmarks):
                    detected_gesture = 'rocked'
                elif is_thumbs_up(hand_landmarks):
                    detected_gesture = 'like'
                elif is_back_hand(hand_landmarks, handedness):
                    detected_gesture = 'goback'
                
        if detected_gesture == 'default' and face_landmarks:
            if is_tongue_out(face_landmarks):
                detected_gesture = 'tongue_out'
            elif is_stare(face_landmarks):
                detected_gesture = 'stare'
                
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
