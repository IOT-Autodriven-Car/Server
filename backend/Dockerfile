FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    pkg-config \
    dos2unix \
    netcat-traditional \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \ 
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libdc1394-22-dev \
    libeigen3-dev \
    libv4l-dev \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip setuptools wheel
RUN pip install --upgrade pip

# Copy the requirements file to the working directory
COPY ./requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
