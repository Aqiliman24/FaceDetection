from flask import Flask, jsonify, request
import cv2
import os
import boto3
from dotenv import load_dotenv
from ulid import ULID
import datetime
import torch
from PIL import Image
# import my Library (Pytorch Framework)
from haroun import ConvPool
from haroun.augmentation import augmentation
from haroun.losses import rmse
import torchvision.transforms as transforms


app = Flask(__name__)

def connectAWS(imgpath):

    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID_FACEDETECTION')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY_FACEDETECTION')
    AWS_BUCKET = os.getenv('AWS_BUCKET_FACEDETECTION')
    AWS_LOCATION = os.getenv('AWS_LOCATION_FACEDETECTION')

    conn = boto3.Session(AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY)


    s3 = conn.resource('s3')

    res = s3.Bucket(AWS_BUCKET).upload_file(imgpath,imgpath,ExtraArgs={ "ContentType": "image/jpeg"})
    print(res)

    if res is None:
        print('File Uploaded Successfully')
    else:
        print('File Not Uploaded')



@app.route("/landing_page", methods=["GET"])
def landing_page():
    return ("""<html>
    <body>
    <head><title>Page Title</title></head>

    <h1>Welcome to Face Recognition Landing Page</h1>
    <p>If you can see this, it means the api is working!.</p>

    </body>
    </html>""")
    

@app.route("/api", methods=["POST"])
def face_detection():
    
    data = request.files.get('image')
    imgpath = 'FailedImage/' + str(ULID.from_datetime(datetime.datetime.now()))+'.jpg'
    # imgpath = 'FailedImage/' + 'images.jpg'
    data.save(imgpath)
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_alt2.xml")
#     model = torch.load('model/model.pth')
#     model = Network()
#     model.eval()
    

#     # Transform the image
#     transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

    # Read the input image
    img = cv2.imread(imgpath)
    img = cv2.resize(img,(500,400))

    # Open and transform the image
    # img2 = Image.open(imgpath)
    # img_tensor = transform(img2).unsqueeze(0)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20))
    # faces2 = model(img_tensor)
    # _, pred = faces2.max(1)
    # print('Detected class label:', pred.item())



    if len(faces) > 0: 
        value=True
        print ("True")
    else:
        value=False
        print ("False")
        connectAWS(imgpath)
    
    cv2.waitKey()
    os.remove(imgpath)
    return jsonify(output=value)

if __name__ == '__main__':
    load_dotenv()
    envport = (os.getenv('PORT_FACEDETECTION'))
    app.run(host = '0.0.0.0',port=envport,debug=True)