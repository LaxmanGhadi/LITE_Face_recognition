import cv2
import numpy as np


from numpy import dot
from numpy.linalg import norm
import numpy as np 
import tensorflow as tf 
import cv2 
import os 
import time
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_TFLITE_DISABLE_XNNPACK"] = "1"
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
    return cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    # return frame[y:y+h, x:x+w]
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
       
# -----------------------------LIVE-----------------------------------------
cap = cv2.VideoCapture(0)
text1 = "Searching Face"
text2 = "Face Found! Stay still"
name = ''
ret, frame = cap.read()
save_path = ''
x = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))/3.5)
y = 50
dir_list = []
i =1 
while True:
  ret, frame = cap.read()
  ret2,frame2 = cap.read()
  if ret==False:
    continue
  else:
    face = detect_face(frame)
    cv2.imshow("frame",frame)
    if face is not None:
       cv2.putText(frame, text2, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 66, 111), 2, cv2.LINE_AA)
       if len(name) == 0:
            name = input("Enter the name of this person")
       if len(name)>0:
          save_path = os.path.join('SiameseDataset\Original Images\Original Images',name)
          os.makedirs(save_path, exist_ok=True)
          
          if cv2.waitKey(1) & 0xFF == ord('a'):
        #   for i in range(6):
            img_name = os.path.join(save_path,f"{name}_{i}.jpg")
            cv2.imwrite(img_name,frame2)
            print(f"Taking {i} image of {name}")
            i+=1
    else:
       cv2.putText(frame, text1, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (212, 245, 66), 2, cv2.LINE_AA)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quitting stream...")
        break
cap.release()
cv2.destroyAllWindows()

# name = save_path.split("\\")[-1]
# embeddings = []
# person_database = {}
# for img_name in os.listdir(save_path):

#     local_face = detect_face( cv2.imread(os.path.join(save_path,img_name)))
#     if local_face is not None:
#         pic_embed = get_embed(local_face)
#         if pic_embed is not None and pic_embed.shape == (512,):
#             embeddings.append(pic_embed)
#     if len(embeddings) >0 :
#        person_database[name] = np.array(embeddings)

# global_database = np.load("Face_embedding.npz")
# global_dict = {key: global_database[key] for key in global_database.files}
# global_database.close()
# database = {**global_dict, **person_database}
# np.savez("Face_embedding.npz", **database)
# print(f"{name}'s  encoding has been added")




