from ubuntu:18.04

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt update -y
RUN apt upgrade -y
RUN apt install python3-pip -y
RUN pip3 install spacy
RUN python3 -m spacy download en_core_web_sm

COPY . /QA
WORKDIR /QA

CMD chmod 777 ask
CMD chmod a+x ask
CMD chmod 777 answer
CMD chmod a+x answer

ENTRYPOINT ["/bin/bash", "-c"]

