from numpy import dot
from numpy.linalg import norm
import numpy as np 
import tensorflow as tf
import cv2 
import os
import time
from vars import Embeddings ,Embed_size, Embedder 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
os.environ["TF_TFLITE_DISABLE_XNNPACK"] = "1"
path = Embedder
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
database = np.load(Embeddings)
os.makedirs('Dataset/People/',exist_ok=True)
persons_dir = 'Dataset/People/'

pt = 0

def detect_face(frame):
  face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
  faces = face_cascade.detectMultiScale(frame,1.05,5)
  if len(faces) > 0:
    x,y,h,w  = faces[0]
    if h*w > 28000: 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        return frame[y:y+h,x:x+w]
    # return frame
  else :
    return None

def preprocess(face_img, size=(160, 160)):
    if face_img is None:
        print("⚠️ preprocess() received None image.")
        return None
    if not isinstance(face_img, np.ndarray):
        print("⚠️ preprocess() received non-array input:", type(face_img))
        return None
    if face_img.size == 0:
        print("⚠️ preprocess() received empty image.")
        return None
    img = cv2.resize(face_img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def get_embed(face_img):
    preprocessed_img = preprocess(face_img)
    if preprocessed_img is None:
        print("❌ Skipping embedding due to invalid preprocessing result.")
        return None
    # Check model input expectations
    expected_dtype = input_details[0]['dtype']
    # Ensure dtype and shape match
    preprocessed_img = preprocessed_img.astype(expected_dtype)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
    interpreter.invoke()
    emb = interpreter.get_tensor(output_details[0]['index'])
    # Normalize safely
    if np.linalg.norm(emb) == 0:
        print("⚠️ Zero norm embedding, skipping normalization.")
        return emb[0]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb[0]


def cos_similarity(embed1, embed2):
  return dot(embed1, embed2)/(norm(embed1)*norm(embed2))

def check_person(live_frame):
    live_embed = get_embed(live_frame)
    best_score = -1
    best_name = "Unknown"

    for name, emb_list in database.items():
        # emb_list is a 2D array: (N, 512)
        similarities = [cos_similarity(live_embed, emb) for emb in emb_list]
        avg_score = np.mean(similarities)

        if avg_score > best_score:
            best_score = avg_score
            best_name = name

    return best_score, best_name

# -----------------------------LIVE-----------------------------------------
# cap = cv2.VideoCapture(0)
# while True:
#   ret, frame = cap.read()
#   if ret==False:
#     continue
#   else:
#     face = detect_face(frame)
    
#     if face is None:
#        cv2.putText(frame, "No face in the frame",(50,50),cv2.FONT_HERSHEY_DUPLEX , 1, (245, 66, 111), 2, cv2.LINE_AA)
#     else:
#        Score,Name = check_person(face)
#        cv2.putText(frame, Name +str(Score*100)+"%",(50,50),cv2.FONT_HERSHEY_DUPLEX , 1, (245, 66, 111), 2, cv2.LINE_AA)
#        #On press person check
#         # cv2.imshow("frame",face)
#         # if cv2.waitKey(1) & 0xFF == ord('a'):
#         #   person,Name = check_person(face)
#         #   print(person,Name)
#         # #   break
#     cv2.imshow('frame',frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("👋 Quitting stream...")
#         break
# cap.release()
# cv2.destroyAllWindows()

       




