import shutil
import cv2
import mediapipe as mp
from werkzeug.utils import secure_filename
import tensorflow as tf
import os
from flask import Flask, jsonify, request, flash, redirect, url_for
from pyngrok import ngrok
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import subprocess
# #-----------------------------------------------------
# app = Flask(__name__, template_folder='./templates')
app = FastAPI()
# #-----------------------------------------------------
#
#
# #-----------------------------------------------------
# Tempat deklarasi variabel-variabel penting
filepath = ""
list_class = ['Diamond','Oblong','Oval','Round','Square','Triangle']
list_folder = ['Training', 'Testing']
face_crop_img = True
face_landmark_img = True
landmark_extraction_img = True
# #-----------------------------------------------------
#
#
# #-----------------------------------------------------
# Tempat deklarasi model dan sejenisnya
selected_model = tf.keras.models.load_model(f'models/fc_model_1.h5', compile=False)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# #-----------------------------------------------------
#
#
# #-----------------------------------------------------
# Tempat setting server
UPLOAD_FOLDER = './upload'
UPLOAD_MODEL = './models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','zip','h5'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['UPLOAD_MODEL'] = UPLOAD_MODEL
# app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
# #-----------------------------------------------------

#
from file_processing import FileProcess
from get_load_data import GetLoadData
from data_preprocess import DataProcessing
from train_pred import TrainPred
#
# #-----------------------------------------------------
#
data_processor = DataProcessing()
data_train_pred = TrainPred()
#
import random
def preprocessing(filepath):
    folder_path = './static/temporary'

    shutil.rmtree(folder_path)
    os.mkdir(folder_path)

    data_processor.detect_landmark(data_processor.face_cropping_pred(filepath))

    files = os.listdir(folder_path)
    index = 0
    for file_name in files:
        # Mendapatkan ekstensi file
        file_ext = os.path.splitext(file_name)[1]
        # Membuat nama file acak dengan urutan bilangan acak dari 1-100000
        new_file_name = str(index) + "_" + str(random.randint(1, 100000)) + file_ext
        # Rename file
        os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))
        index += 1

    print("Tungu sampai selesaiii")

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.)

#-----------------------------------------------------
# Fungsi untuk menjalankan ngrok
def run_ngrok():
    try:
        # Jalankan ngrok dan simpan prosesnya
        ngrok_process = subprocess.Popen(['ngrok', 'http', '8000'])
        return ngrok_process
    except Exception as e:
        print(f"Error running ngrok: {e}")


@app.get("/")
async def root():
    # Dapatkan URL publik dari ngrok
    ngrok_url = "Tidak Ada URL Publik (ngrok belum selesai memulai)"
    try:
        ngrok_url = subprocess.check_output(['ngrok', 'http', '8000']).decode('utf-8').strip()
    except Exception as e:
        print(f"Error getting ngrok URL: {e}")

    return {"message": "Hello, World!", "ngrok_url": ngrok_url}

# def get_models():
#     folder_path = './models/'
#     files = os.listdir(folder_path)
#     count = len(files)
#     return {'count': count}
#
# # @app.delete('/delete_img')
# # def delete_img():
# #     for i in range (0,4):
# #         if os.path.exists(f"./static/result_upload{i}.jpg"):
# #             os.remove(f"./static/result_upload{i}.jpg")
# #             print("File terhapus")
# #             return jsonify({'message': 'Berhasil di hapus'}), 400
# #         else:
# #             print("File tidak ditemukan.")
# #             return jsonify({'message': 'No file selected for uploading'}), 400
# #


## -------------------------------------------------------------------------
##                   API UNTUK MELAKUKAN PROSES PREDIKSI
## -------------------------------------------------------------------------

@app.post('/upload/file',tags=["Predicting"])
async def upload_file(picture: UploadFile):
    file_extension = picture.filename.split('.')[-1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail='Invalid file extension')

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(picture.filename))
    with open(file_path, 'wb') as f:
        f.write(picture.file.read())
    try:
        print("cell")
        processed_img = preprocessing(cv2.imread(file_path))
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f'Error processing image: {str(e)}')

    print(processed_img)
    if processed_img is not None:
        _, img_encoded = cv2.imencode('.png', processed_img)
        img_base64 = img_encoded.tobytes()
    else:
        img_base64 = None

    return JSONResponse(content={'message': 'File successfully uploaded', 'processed_image_base64': img_base64}, status_code=200)

@app.get('/get_images', tags=["Predicting"])
def get_images():
    folder_path = "./static/temporary/"
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    urls = []
    for i in range(0, 4):
        url = f'{public_url}/static/temporary/{files[i]}'
        urls.append(url)
        bentuk, persentase = data_train_pred.prediction(selected_model)
    return {'urls': urls, 'bentuk_wajah':bentuk[0], 'persen':persentase}


## -------------------------------------------------------------------------
##                   API UNTUK MELAKUKAN PROSES TRAINING
## -------------------------------------------------------------------------

@app.post('/upload/dataset', tags=["Training"])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']
    print(file.filename)

    if file.filename == '':
        print('setidaknya sampe sini1')
        return jsonify({'message': 'No file selected for uploading'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    FileProcess.extract_zip(filepath)

    return {'message': 'File successfully uploaded'}

@app.post('/set_params', tags=["Training"])
def set_params():
    global optimizer, epoch, batch_size
    data = request.json

    optimizer = data.get('optimizer')
    epoch = data.get('epoch')
    batch_size = data.get('batchSize')

    response = {'message': 'Preprocessing sukses'}
    return response

@app.get('/get_info_data', tags=["Training"])
def get_info_prepro():
    global optimizer, epoch, batch_size
    training_counts = GetLoadData.get_training_file_counts().json
    testing_counts = GetLoadData.get_testing_file_counts().json
    response = {
        "optimizer": optimizer,
        "epoch": epoch,
        "batch_size": batch_size,
        "training_counts": training_counts,
        "testing_counts": testing_counts
    }
    return response

@app.get('/get_images_preprocess', tags=["Training"])
def get_random_images_crop():
    images_face_landmark = GetLoadData.get_random_images(tahap="Face Landmark",public_url=public_url)
    images_face_extraction = GetLoadData.get_random_images(tahap="landmark Extraction", public_url=public_url)

    response = {
        "face_landmark": images_face_landmark,
        "landmark_extraction": images_face_extraction
    }
    return response

@app.get('/do_preprocessing', tags=["Training"])
async def do_preprocessing():
    try:
        data_train_pred.do_pre1(test="")
        data_train_pred.do_pre2(test="")
        return {'message': 'Preprocessing sukses'}
    except Exception as e:
        # Tangani kesalahan dan kembalikan respons kesalahan
        error_message = f'Error during preprocessing: {str(e)}'
        raise HTTPException(status_code=500, detail=error_message)

@app.get('/do_training', tags=["Training"])
def do_training():
    global epoch
    folder = ""
    if (face_landmark_img == True and landmark_extraction_img == True):
        folder = "Landmark Extraction"
    elif (face_landmark_img == True and landmark_extraction_img == False):
        folder = "Face Landmark"
    # --------------------------------------------------------------
    train_dataset_path = f"./static/dataset/{folder}/Training/"
    test_dataset_path = f"./static/dataset/{folder}/Testing/"

    train_image_df, test_image_df = GetLoadData.load_image_dataset(train_dataset_path, test_dataset_path)

    train_gen, test_gen = data_train_pred.data_configuration(train_image_df, test_image_df)
    model = data_train_pred.model_architecture()

    result = data_train_pred.train_model(model, train_gen, test_gen, epoch)

    # Mengambil nilai akurasi training dan validation dari objek result
    train_acc = result.history['accuracy'][-1]
    val_acc = result.history['val_accuracy'][-1]

    # Plot accuracy
    data_train_pred.plot_accuracy(result=result, epoch=epoch)
    acc_url = f'{public_url}/static/accuracy_plot.png'

    # Plot loss
    data_train_pred.plot_loss(result=result, epoch=epoch)
    loss_url = f'{public_url}/static/loss_plot.png'

    # Confusion Matrix
    data_train_pred.plot_confusion_matrix(model, test_gen)
    conf_url = f'{public_url}/static/confusion_matrix.png'

    return jsonify({'train_acc': train_acc, 'val_acc': val_acc, 'plot_acc': acc_url, 'plot_loss':loss_url,'conf':conf_url})


## -------------------------------------------------------------------------
##                   API UNTUK PEMILIHAN MODEL
## -------------------------------------------------------------------------


@app.post('/upload/model', tags=["Model"])
def upload_model():
    if 'file' not in request.files:
        return {'message': 'No file part in the request'}, 400

    file = request.files['file']

    if file.filename == '':
        return {'message': 'No file selected for uploading'}, 400

    if file and FileProcess.allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_MODEL'], filename)
        file.save(filepath)

        return {'message': 'File successfully uploaded'}

    return {'message': 'File failed to uploaded'}

@app.post('/selected_models')
def select_models(index: int):
    global selected_model
    try:
        global selected_model
        selected_model = tf.keras.models.load_model(f'models/fc_model_{index}.h5')

        # Lakukan sesuatu dengan indeks yang diterima

        return {'message': 'Request berhasil diterima'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error: {str(e)}')


# @app.get('/progress', methods=['GET'])
# def get_progress():
#     global progres, name
#     progres, name = data_train_pred.get_progress_1()
#
#     return jsonify({'progress': progres, 'name': name})
#
# @app.get('/progress2', methods=['GET'])
# def get_progress_2():
#     global prepro_img
#     return jsonify({'progress': prepro_img, 'name': "Landmark Extraction"})


if __name__ == '__main__':
    import uvicorn
    public_url = ngrok.connect(8080).public_url
    print(f' * Running on {public_url}')
    uvicorn.run(app, host="0.0.0.0", port=8080)
    # app = FastAPI()
