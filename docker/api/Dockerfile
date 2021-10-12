FROM python:3.9

ADD . /home/ccs4dt/

WORKDIR /home/ccs4dt

RUN pip install -r requirements.txt -e .

WORKDIR /home/ccs4dt

# FLASK_APP and FLASK_ENV are read from ENV variables
CMD ["sh", "-c", "flask run --host $FLASK_HOST --port $FLASK_PORT"]