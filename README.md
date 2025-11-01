# Face Recognition Project (Using FaceNet)

## 📘 How to Use This Project

### Step 1: Download the Model
1. Go to the following link:  
   👉 [FaceNet Model Assets](https://github.com/shubham0204/FaceRecognition_With_FaceNet_Android/tree/master/app/src/main/assets)
2. Download the **`facenet_512.tflite`** model.
3. Add the downloaded model to the **same directory** as this project.

> **Note:**  
> - `facenet_512.tflite` → 512-size encoding (default)
> - There’s also a `facenet.tflite` model (128-size encoding) which is faster. See [Alternative Model Option](#alternative-model-option) below.

---

### Step 2: Add a New Face
1. Run **`Add_new_face.py`**  
2. Enter the **name of the person** whose images you want to take.
3. The camera will start. Stay in front of it until it detects your face (bounding box appears).
4. Press **`A`** to take a photo.  
   - Take **at least 6 photos** for best results.
   - After each capture, the program will display how many photos have been taken.

---

### Step 3: Encode Faces
1. Once photos are taken, go to **`Face_All_Face_Encode.py`**.
2. Run this file — it will encode all faces present in the `face` folder.

> ⚠️ **Important Note:**  
> Even if you add **only one new person**, you **must re-encode all photos**.  
> Loading the model at different times produces slightly different encodings, which can cause mismatches with preexisting encodings.

3. After encoding, the program will confirm by displaying:  
   `"person_name faces are encoded"`

---

### Step 4: Run Face Recognition
Run **`Face_recog_V2.py`** to perform **face recognition** on live camera input.

---

## ⚙️ Alternative Model Option

If you prefer to use the **128-dimension encoding model** (`facenet.tflite`), follow these steps:

1. Download **`facenet.tflite`** and place it in the same directory as this project.
2. Open the following files and update the model name:
   - `Add_new_face.py`
   - `All_Face_Encod.py`
   - `Face_recog_V2.py`
3. In **`All_Face_Encod.py`**, go to **line 77** and change the 512 to 128:
   ```python 
   if pic_embed is not None and pic_embed.shape == (512,):
