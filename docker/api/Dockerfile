FROM python:3.9

##################################################
## Create and use a user to match the host user ##
##################################################
ARG UID
ARG GID
ARG USER

RUN addgroup --system $USER --gid "$GID" &&\
    adduser --disabled-password --gecos "" --home "/home/$USER" --ingroup "$USER" --no-create-home --uid "$UID" "$USER" && \
    install -d -m 0755 -o $USER -g $USER "/home/$USER"

##################################################
## Add code to container                        ##
##################################################
ADD . /home/ccs4dt/

WORKDIR /home/ccs4dt

##################################################
## Install dependencies                         ##
##################################################
RUN pip install -r requirements.txt -e .

WORKDIR /home/ccs4dt

##################################################
## FLASK_APP and FLASK_ENV are read from ENV variables
##################################################

## Switch to host user
USER ${USER}

CMD ["sh", "-c", "flask run --host $FLASK_HOST --port $FLASK_PORT"]