FROM ubuntu:latest
ENV TZ=Asia/Kolkata \
    DEBIAN_FRONTEND=noninteractive

RUN set -xe \
    && apt-get update -y\
    && apt-get upgrade -y\	
    && apt-get install -y python3\
    && apt-get install -y python3-pip\
    && apt-get install -y build-essential\
    && apt-get install -y python3-opencv
RUN pip install --upgrade pip

COPY . .

WORKDIR .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]