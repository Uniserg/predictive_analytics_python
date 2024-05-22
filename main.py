from flask import Flask, request, jsonify
import flask

import category_encoders
import numpy as np
import pickle
import pandas as pd

# Загрузка модели
from keras.models import load_model

app = Flask(__name__)


home = 'C:/Users/sergi/PycharmProjects/PythonEnv/Предиктивная аналитика/'

# model = load_model(f'{home}/model.h5')

# Загрузка модели с помощью pickle
with open(f'{home}/scaler_categorial.pkl', 'rb') as file:
    scaler_categorial = pickle.load(file)

# Загрузка модели с помощью pickle
with open(f'{home}/scaler_number.pkl', 'rb') as file:
    scaler_number = pickle.load(file)

# Загрузка модели с помощью pickle
with open(f'{home}/model.pkl', 'rb') as file:
    model = pickle.load(file)

order_x = {'vehicleType': 1, 
 'yearOfRegistration': 2, 
 'gearbox': 3, 
 'powerPS': 4,
 'model':5, 
 'kilometer': 6, 
 'fuelType':7, 
 'brand':8, 
 'notRepairedDamage':9
}

def get_cols(df) -> list:
    """
    Функция возвращает список категориальных и числовых переменных.
    """
    categorical_feature_mask = df.dtypes == object
    number_feature_mask = df.dtypes != object
    numbers_cols = df.columns[number_feature_mask].tolist()
    categorical_cols = df.columns[categorical_feature_mask].tolist()
    return numbers_cols, categorical_cols

def get_vector_x(data: dict):
    df = pd.DataFrame([data])
    print(df)
    numbers, categorical = get_cols(df)
    df[categorical] = scaler_categorial.transform(df[categorical])
    df[numbers] = scaler_number.transform(df[numbers])
    return np.array(list(map(lambda x: df[x].values[0], df.columns))).reshape(1, -1)

def predict_price(car_data):
    return model.predict(get_vector_x(car_data))[0]


@app.route('/api/get-car-price', methods=['POST'])
def get_car_price():
    request_data = request.get_json()
    if not request_data:
        return jsonify({'error': 'No data provided in the request body'}), 400

    request_data['notRepairedDamage'] = 'nein' # TODO: хардкод

    print(request_data)

    if 'price' in request_data:
        del request_data['price']

    price = predict_price(request_data)
    
    print("type =", type(price))
    if type(price) is not np.float32 and type(price) is not np.float64:
        price = price[0]
    
    print(f'predicted price = {price}')
    response = flask.jsonify({'price': str(price)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    # Here you can add the car object to your database or any other storage
    # return jsonify({'price': car.price}), 200
    return response

if __name__ == '__main__':
    app.run(debug=True)
