FROM nvcr.io/nvidia/tritonserver:25.06-py3

RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get install -y ffmpeg

RUN pip install --no-cache-dir transformers==4.38.2 accelerate==0.22.0
RUN pip install --no-cache-dir torch==2.2.1
RUN pip install --no-cache-dir torchaudio

