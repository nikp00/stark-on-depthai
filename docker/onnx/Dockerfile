FROM ubuntu:20.04
RUN apt-get update && apt-get upgrade
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install python3 python3-pip -y
RUN apt-get install ffmpeg libsm6 libxext6 wget cmake -y

RUN pip3 install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt