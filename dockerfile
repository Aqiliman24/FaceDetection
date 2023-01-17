FROM python:3.10.6
WORKDIR /Face image
COPY requirements.txt ./
RUN requirements.txt
COPY . .
CMD ["python","faceDetection.py"]

# FROM nginx
# RUN rm /etc/nginx/conf.d/default.conf
# COPY config/nginx.config /etc/nginx/conf.d/default.conf
# COPY dist/McReportGenerator /usr/share/nginx/html