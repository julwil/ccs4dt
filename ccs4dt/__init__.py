from flask import Flask

app = Flask(__name__)

from ccs4dt.main.http import space_controller, input_batch_controller, output_batch_controller
