from flask import Flask, jsonify, request
import cv2
import os
import boto3
from dotenv import load_dotenv
from ulid import ULID
import datetime


app = Flask(__name__)

# def connectAWS(imgpath):

#     AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID_FACEDETECTION')
#     AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY_FACEDETECTION')
#     AWS_BUCKET = os.getenv('AWS_BUCKET_FACEDETECTION')
#     AWS_LOCATION = os.getenv('AWS_LOCATION_FACEDETECTION')

#     conn = boto3.Session(AWS_ACCESS_KEY_ID,
#             AWS_SECRET_ACCESS_KEY)


#     # bucket = conn.get_bucket(AWS_BUCKET)
#     # conn.create_bucket(Bucket=AWS_BUCKET, CreateBucketConfiguration={
#     # 'LocationConstraint': AWS_LOCATION})

#     s3 = conn.resource('s3')

#     # object = s3.Object(AWS_BUCKET, 'test')

#     res = s3.Bucket(AWS_BUCKET).upload_file(imgpath,imgpath,ExtraArgs={ "ContentType": "image/jpeg"})
#     print(res)
#     # response = conn.upload_file(img, AWS_BUCKET)

#     # res = response.get('ResponseMetadata')

#     if res == ():
#         print('File Uploaded Successfully')
#     else:
#         print('File Not Uploaded')



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
    # imgpath = 'FailedImage/' + str(ULID.from_datetime(datetime.datetime.now()))+'.jpg'
    imgpath = 'FailedImage/' + 'images.jpg'
    data.save(imgpath)
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_alt2.xml")

    # Read the input image
    img = cv2.imread(imgpath)
    img = cv2.resize(img,(500,400))

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20))

    if len(faces) > 0: 
        value=True
        print ("True")
    else:
        value=False
        print ("False")
        # connectAWS(imgpath)
    
    # os.remove(imgpath)
    

    cv2.waitKey()
    return jsonify(output=value)

if __name__ == '__main__':
    load_dotenv()
    # test = (os.getenv('PORT_FACEDETECTION'))
    app.run(host = '0.0.0.0',port=8002,debug=True)