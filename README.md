# EPI-detector

Openvino custom trained PPE yellow jacket detector

## Pre-requisites

* Python 3.6+
* OpenVino

```
python -m pip install -r requirements.txt
```

## Usage

Start inference by using

```
python Interface.py -m [person detection model path.xml] -c [classification model path.xml] -i [iput video source path] -d [CPU, GPU, MYRIAD]
```
