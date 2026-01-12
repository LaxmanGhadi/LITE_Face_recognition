
import numpy as np 
import tensorflow as tf 
import cv2 
import os 
from Functions import  crop_face  ,get_embed, persons_dir
import time 
pt = time.time()

def create_embedding_database():
  persons_list=os.listdir(persons_dir)
  databse = {}
  for person in persons_list:
    person_pics = os.listdir(os.path.join(persons_dir,person))
    embeddings = []
    for i in range(min(6,len(persons_list))):
      local_face = crop_face(cv2.imread(os.path.join(os.path.join(persons_dir,person),person_pics[i])))
      if local_face is not None:
        pic_embed = get_embed(local_face)
        if pic_embed is not None and pic_embed.shape == (512,):
            embeddings.append(pic_embed)
    if len(embeddings) >0 :
       databse[person] = np.array(embeddings)
    else:
       print(f'No image received for {person}')
  np.savez('Face_embedding_512.npz', **databse)
  print(f"\n🎯 Embedding database saved: {len(databse)} people total")
  print(time.time()-pt)





       




