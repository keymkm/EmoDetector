import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input

import numpy as np


class EmoClassifier:
    def __init__(self, p_model_path, p_base_model, p_emo_classes_names):
        '''
        :argument
        p_model_path - путь к сохраненой модели классификатора эмоций в формате h5
        p_base_model - название базовой модели, возможные значения:
        - resnet50,
        - vgg16,
        - xception

        p_emo_classes_names - список названий классов эмоций
        '''

        assert p_base_model in ['resnet50', 'vgg16', 'xception'], 'Неизвестный тип базовой одели "{}"'.format(
            p_base_model)
        self.base_model_name = p_base_model

        # Размер изображения, кот принимает на вход модель
        self.image_size = 224

        # функция преобразования изображения для подачи в модель
        self.preprocess_input_func = None
        if self.base_model_name == 'resnet50':
            self.preprocess_input_func = resnet50_preprocess_input
        elif self.base_model_name == 'vgg16':
            self.preprocess_input_func = vgg16_preprocess_input
        elif self.base_model_name == 'xception':
            self.preprocess_input_func = xception_preprocess_input
            self.image_size = 299
        # elif

        # модель классификатора эмоций
        self.model = tf.keras.models.load_model(p_model_path)

        # список названий классоф эмоций
        self.emo_classes_names = p_emo_classes_names.copy()

    # init

    def predict(self, p_face_img):
        '''
        Получение значения эмоции по тизображению лица
        p_face_img - изображение лица предобработанное для загрузки в модель

        :return
        название предсказанной эмоции
        '''

        predictions = self.model.predict(p_face_img)
        return self.emo_classes_names[np.argmax(predictions)]
    # predict

# EmoClassifier