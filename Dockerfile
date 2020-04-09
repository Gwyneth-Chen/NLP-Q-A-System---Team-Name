from ubuntu:18.04

RUN apt update -y
RUN apt upgrade -y
RUN apt install python3-pip -y

COPY . /QA
WORKDIR /QA

CMD chmod 777 ask
CMD chmod a+x ask
CMD chmod 777 answer
CMD chmod a+x answer

ENTRYPOINT ["/bin/bash", "-c"]

