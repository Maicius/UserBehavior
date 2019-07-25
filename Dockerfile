FROM python:3.6

MAINTAINER maicius
WORKDIR /UserBehavior
COPY requirements.txt /UserBehavior
RUN pip install -i https://mirrors.nju.edu.cn/pypi/web/simple/ -r requirements.txt
COPY . /UserBehavior
CMD python3 src/run_dynamic.py
