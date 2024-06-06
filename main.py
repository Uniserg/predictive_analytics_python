import flask

import category_encoders
import numpy as np
import pickle
import pandas as pd

from keras.models import load_model

from lab1 import lab1
from lab2 import lab2

app = flask.Flask(__name__)

# lab1
@app.route('/api/lab1/get-car-price', methods=['POST'])
def get_car_price():
    request_data = flask.request.get_json()
    if not request_data:
        return flask.jsonify({'error': 'No data provided in the request body'}), 400

    price = lab1.get_car_price(request_data)

    response = flask.jsonify({'price': str(price)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


#lab2
@app.route('/api/lab2/get-interest-by-user-id', methods=['GET'])
def get_interest_by_user_id():
     # Получение query parameter
    user_id = flask.request.args.get('user_id')

    if user_id is None:
        # Возвращение ошибки, если query parameter не указан
        return flask.jsonify({'error': 'Missing query parameter: user_id'}), 400
    
    predict = lab2.get_interest_by_user_id(user_id)
    response = flask.jsonify({'interest': predict})
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/api/lab2/get-user-by-interest', methods=['GET'])
def get_users_by_interest():
    interest = flask.request.args.get('interest')
    response = flask.jsonify(lab2.get_users_by_interest(interest))
    return response


if __name__ == '__main__':
    app.run(debug=True)
