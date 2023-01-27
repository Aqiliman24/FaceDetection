from flask import Flask, jsonify, request
# from sklearn.metrics import accuracy_score
import cv2

app = Flask(__name__)

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
    data.save("data/image.jpg")
    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier("model/haarcascade_frontalface_alt.xml")

    # Read the input image
    img = cv2.imread('data/image.jpg')
    img = cv2.resize(img,(500,400))

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0: 
        value=True
        print ("True")
    else:
        value=False
        print ("False")

    cv2.waitKey()
    return jsonify(output=value)

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=8002,debug=True)
