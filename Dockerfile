FROM tensorflow/tensorflow:2.0.0b1-gpu-py3

MAINTAINER maicius
WORKDIR /UserBehavior
COPY requirements.txt /UserBehavior
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
COPY . /UserBehavior
CMD python3 src/run_dynamic.py
