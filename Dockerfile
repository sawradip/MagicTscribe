# Start with an NVIDIA CUDA image
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

# Set non-interactive frontend (useful to avoid interactive prompts during package installations)
ENV DEBIAN_FRONTEND=noninteractive

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

# Install Python 3.9
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3.9-venv python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --set python3 /usr/bin/python3.9

# Update pip and install wheel
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install wheel


RUN apt-get -y update && apt-get -y upgrade && apt-get install -y git ffmpeg

WORKDIR /src
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r ./requirements.txt

COPY . /src

EXPOSE 7860

CMD [ "python3", "main.py"]
# CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9307" ]