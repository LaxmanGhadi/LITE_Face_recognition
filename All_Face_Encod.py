from numpy import dot
from numpy.linalg import norm
import numpy as np 
import tensorflow as tf 
import cv2 
import os 
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_TFLITE_DISABLE_XNNPACK"] = "1"
path = 'facenet80M.tflite'
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def detect_face(frame):
  face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
  faces = face_cascade.detectMultiScale(frame,1.05,5)
  if len(faces) > 0:
    x,y,h,w  = faces[0]
    # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    return frame[y:y+h, x:x+w]
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

def create_embedding_database():
  persons_dir = 'SiameseDataset/Original Images/Original Images'
  persons_list=os.listdir(persons_dir)
  databse = {}
  for person in persons_list:
    person_pics = os.listdir(os.path.join(persons_dir,person))
    embeddings = []
    # check every pic 
    # for person_pic in person_pics:
    #   cv2.imread(os.path.join(os.path.join(persons_dir,person),person_pic))
    for i in range(6):
      local_face = detect_face(cv2.imread(os.path.join(os.path.join(persons_dir,person),person_pics[i])))
      if local_face is not None:
        pic_embed = get_embed(local_face)
        if pic_embed is not None and pic_embed.shape == (512,):
            embeddings.append(pic_embed)
    if len(embeddings) >0 :
       databse[person] = np.array(embeddings)
    else:
       print(f'No image received for {person}')
  np.savez("Face_embedding.npz", **databse)
  print(f"\n🎯 Embedding database saved: {len(databse)} people total")
    

create_embedding_database()
#   Saving Score Here


       




