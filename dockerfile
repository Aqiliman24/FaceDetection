FROM python:3.10.6
WORKDIR /faceimagerecog
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY . .
CMD ["python","face_detection.py"]

# FROM nginx
# RUN rm /etc/nginx/conf.d/default.conf
# COPY config/nginx.config /etc/nginx/conf.d/default.conf
# COPY dist/McReportGenerator /usr/share/nginx/html