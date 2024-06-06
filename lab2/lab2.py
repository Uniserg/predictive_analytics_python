from flask import Flask, request, jsonify
import flask

import category_encoders
import numpy as np
import pickle
import pandas as pd
import time
import json
import re
import vk_api
# from request_utils import *


app = Flask(__name__)


home = 'C:/Users/sergi/PycharmProjects/PythonEnv/Предиктивная аналитика/lab2'


with open(f"{home}/vk_token.txt") as f:
   vk_token = f.read()

session = vk_api.VkApi(token=vk_token)
vk = session.get_api()

version = 5.199

fields_name = ("id,first_name,last_name,sex,bdate,country,home_town,"
    "has_mobile,followers_count,schools,common_count,universities,personal")

def get_profile_infos_by_user_ids(user_ids, fields_name):
  batch_size = 1000
  k = 0
  data = []

  while k < len(user_ids):
    data.extend(vk.users.get(
        user_ids=",".join(map(str, user_ids[k:k + batch_size])),
        v=version,
        fields=fields_name
        ))
    k += batch_size
    
  return data


def extract_text_data(json_string):
    """
    Извлекает и объединяет текстовые данные из полей 'name', 'description' и 'activity' в JSON-строке.
    """
    if json_string == "[]":
        return ""
    try:
        data = pd.DataFrame(json.loads(json_string))
        pattern = r"(\w*[ЦцКкНнГгШшЩщЗзХхФфВвПпРрЛлДдЖжЧчСсМмТтБб])[УуЕеЫыАаОоЭэЯяИиЬьЮюЯяЙй]*"
        for col in ['name', 'description', 'activity']:
            data[col] = data[col].fillna("").apply(
                lambda x: " ".join(
                    x.lower() for x in re.findall(pattern, x)
                    )
                )
        data['combined'] = data['name'] + " " + data['description'] + " " + data['activity']
        return data['combined'].str.strip().sum()
    except:
        return ""
  


def normalize_text(column):
    """
    Нормализует текст в столбце, приводя его к нижнему регистру и извлекая слова с кириллическими символами.
    """
    if not isinstance(column, str):
      column = ''

    return " ".join(
        x.lower() for x in re.findall(
            r"(\w*[ЦцКкНнГгШшЩщЗзХхФфВвПпРрЛлДдЖжЧчСсМмТтБб])[УуЕеЫыАаОоЭэЯяИиЬьЮюЯяЙй]*",
            column)
    )


def collect_group_info(users):
  df = pd.DataFrame(users)

  df["Группы"] = None

  for index, row in df.iterrows():
      try:
          groups = vk.groups.get(
              user_id=row['id'],
              v=version,
              extended=1,
              fields="description,activity"
              )
          df.at[index, "Группы"] = json.dumps(groups["items"])
      except Exception as e:
          df.at[index, "Группы"] = None
          print(f"Ошибка: {e}")
      if (index + 1) % 10 == 0:
          print(f"Обработано {index + 1} профилей")
      time.sleep(0.3)


  df["Подписки"] = None

  for index, row in df.iterrows():
      try:
          subscriptions = vk.users.getSubscriptions(
              user_id=row['id'],
              v=version,
              extended=1,
              fields="description,activity"
          )
          df.at[index, "Подписки"] = json.dumps(subscriptions["items"])
      except Exception as e:
          df.at[index, "Подписки"] = None
          print(f"Ошибка: {e}")
      if (index + 1) % 10 == 0:
          print(f"Обработано {index + 1} профилей")
      time.sleep(0.3)
      
      
  df = df.rename(columns={
    "bdate": "Дата рождения",
    "country": "Страна",
    'has_mobile': "Есть номер",
    'followers_count': "Подписчики",
    'common_count': "Общие друзья",
    'home_town': "Родной город",
    'schools': "Школы",
    'universities': "ВУЗы",
    'sex': "Пол",
    'first_name': "Имя",
    'last_name': "Фамилия",
    'can_access_closed': "Доступ к закрытому профилю",
    'is_closed': "Закрытый профиль",
    'deactivated': "Неактивная страница",
    'personal': "Позиция"
    }
               )
  
  if 'Дата рождения' in df.columns:
    df["Дата рождения"].str.split(".").str.len().value_counts(dropna=False)
    df.loc[df["Дата рождения"].str.split(".").str.len() < 3, "Дата рождения"] = None
    df["Дата рождения"] = df["Дата рождения"].str.extract(r"(\d{4})$")
    df.loc[df["Дата рождения"].isnull(), "Дата рождения"] = None
    df["Дата рождения"] = df["Дата рождения"].astype('float').astype('Int64')
    
  df.drop(columns=[col for col in ['Страна', "Родной город", 'Есть номер', 'Доступ к закрытому профилю', 'Закрытый профиль', "ВУЗы", "Школы"] if col in df.columns], inplace=True)
  
  if "Неактивная страница" in df.columns:
    df.drop(columns=["Неактивная страница"], inplace=True)


  # return df['Позиция'].fillna("{}").replace("[]", "{}")
  if 'Позиция' in df.columns:
    personal_info = df['Позиция'].fillna('{}').astype(str).apply(lambda x: x.replace("'", '"')).apply(json.loads).apply(pd.Series)
    # .apply(lambda x: json.loads(x.replace("'", '"')))
    df = df.join(personal_info)

    df.rename(columns={
        'alcohol': 'Алкоголь',
        'inspired_by': 'Вдохновение',
        'langs': 'Языки',
        'life_main': 'Смысл жизни',
        'people_main': 'Главное в людях',
        'political': 'Политическая позиция',
        'smoking': 'Курение',
        'religion': 'Религия',
        'religion_id': 'id религии'},
              inplace=True
              )
    
    if 'Алкоголь' in df.columns:
        df['Алкоголь'] = df['Алкоголь'].replace(0.0, None).fillna(df['Алкоголь'].mean())


    meanings_of_life = {
      1: "Семья и дети",
      2: "Карьера и деньги",
      3: "Развлечение и отдых",
      4: "Наука и исследования",
      5: "Совершенствование мира",
      6: "Саморазвитие",
      7: "Красота и искусство",
      8: "Слава и влияние"
      }


    values_in_people = {
        1: "Ум и креативность",
        2: "Доброта и честность",
        3: "Красота и здоровье",
        4: "Власть и богатство",
        5: "Смелость и упорство",
        6: "Юмор и жизнелюбие"
        }

    df['Главное в людях'] = df['Главное в людях'].replace(0, None).replace(values_in_people)
    df['Курение'] = df['Курение'].replace(0, None).fillna(df['Курение'].mean())
    df['Смысл жизни'] = df['Смысл жизни'].replace(0, None).replace(meanings_of_life)


    political_positions = {
      1: "Коммунизм",
      2: "Социализм",
      3: "Аполитическая",
      4: "Либерализм",
      5: "Консерватизм",
      6: "Монархия",
      7: "Ультраконсерватизм",
      8: "Аполитическая",
      9: "Либертарианство"
      }


    if 'Политическая позиция' in df.columns:
      df['Политическая позиция'] = df['Политическая позиция'].replace(0, None).replace(political_positions)

  df['Группы'] = df['Группы'].fillna("[]").apply(extract_text_data)
  df['Подписки'] = df['Подписки'].fillna("[]").apply(extract_text_data)

  columns_to_normalize = [col for col in ["Вдохновение", "Смысл жизни", "Главное в людях", "Политическая позиция", "Религия"] if col in df.columns]

  for column in columns_to_normalize:
      df[column] = df[column].apply(normalize_text)


  categorical_text = [txt for txt in ['Группы', 'Подписки', 'Смысл жизни', 'Главное в людях', 'Политическая позиция', 'Религия', 'Вдохновение'] if txt in df.columns]

  df['Текст'] = df[categorical_text].fillna('').agg(' '.join, axis=1)

  df = df.drop(columns=categorical_text)
  
  txt = df["Текст"].sum()
  tdf = pd.Series(txt.split())

  tdf = tdf[tdf != ""]
  tdf = pd.DataFrame(tdf.value_counts()).reset_index().rename(columns={"index": "Слово", 0: "Частота"})
  tdf['Длина'] = tdf['Слово'].str.len()
  tdf = tdf[tdf['Длина'] > 4]
  keywords = ["музык", "москв", "работ", "новост", "фотограф", "творчеств", "реклам", "бизнес", "магазин", "культур"]
  categories = ["Музыка", "Москва", "Работа", "Новости", "Фотография", "Творчество", "Реклама", "Бизнес", "Магазин", "Культура"]

  def calculate_frequencies(text):
      """
      Вычисляет частоту встречаемости ключевых слов в тексте.
      """
      words = pd.Series(text.split())
      filtered_words = words[words != ""]
      word_counts = filtered_words.value_counts().reindex(keywords, fill_value=0)
      return word_counts

  frequencies = df['Текст'].apply(calculate_frequencies)
  df[categories] = frequencies[keywords].values.tolist()
  df = df.drop(columns=["Текст"])

  df = df.set_index("id")
  categories = ["Музыка", "Москва", "Работа", "Новости", "Фотография", "Творчество", "Реклама", "Бизнес", "Магазин", "Культура"]
  sum_categories = df[categories].fillna(0).sum(axis=1)
  df[categories] = df[categories].div(sum_categories, axis=0) * 100
  df[categories] = df[categories].round()

  # df = df.dropna()
  
  return df

with open(f'{home}/scalers/scalers.pkl', 'rb') as file:
    scalers = pickle.load(file)

with open(f'{home}/models/k_means_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open(f'{home}/data/cluters_data.pkl', 'rb') as file:
    data = pickle.load(file)

def scale_features(df):
  res = df.copy()
  for col, scaler in scalers.items():
    res[col] = scaler.transform(res[[col]])
  return res

features = list(data.keys())

    
def get_interest_by_user_id(user_id):
    p = collect_group_info(get_profile_infos_by_user_ids([user_id], fields_name))
    pred = model.predict(scale_features(p[["Музыка", "Москва", "Работа", "Новости", "Фотография", "Творчество", "Реклама", "Бизнес", "Магазин", "Культура"]].fillna(0)))
    print(pred[0], user_id)
    return features[pred[0]]

def get_users_by_interest(interest):
    if interest is None:
        return flask.jsonify({'error': 'Missing query parameter: interest'}), 400
    
    if interest not in data:
        return jsonify({'error': f'В базе нет интереса: {interest}'}), 404
    
    json_objects = data[interest].to_dict(orient='records')
    
    return json_objects

if __name__ == '__main__':
    app.run(debug=True)
