import configparser
import os
import emo_classificator_model as ecm
import cv2
import numpy as np

#Путь к файлу настроек
CONFIG_PATH = 'emo_detector.conf'

class EmoDetector:
    def __init__(self):
        '''Инициализация праметров'''

        #Загружаем параметры
        assert os.path.exists(CONFIG_PATH), 'Не найден файл конфигурации "{}"'.format(CONFIG_PATH)

        config = configparser.ConfigParser()
        config.read (CONFIG_PATH)

        # путь к файлу с моделью классификатора эмоций
        self.model_path = config['emo_classificator']['model_path']
        # название базовой модели (архитектуры), на кот. построен классификатор
        self.base_model = config['emo_classificator']['base_model']

        # загрузим названия эмоций (классов)
        class_names_path = config['emo_classificator']['class_names_path']

        assert os.path.exists(class_names_path), 'Не найден файл c названиями классов эмоций "{}"'.format(class_names_path)

        self.emo_classes_names = []
        with open(class_names_path, 'r') as f:
            self.emo_classes_names = f.read().split(',')

        #Загрузим модель
        self.emo_classifier = ecm.EmoClassifier(self.model_path, self.base_model, self.emo_classes_names)

        #Настройки opencv
        #Классификатор для распознования лиц
        face_cascade_path = config['opencv']['haarcascade_path']
        assert os.path.exists(face_cascade_path), 'Не найден файл c классификатора распозования лиц "{}"'.format(face_cascade_path)

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    #__init__

    def run(self, p_cam_num = 0, p_freq_emo_class = 5, p_frame_color  = (0, 255, 0), p_label_color = (0, 255, 0), p_font_scale = 0.7):
        '''
        Запуск детекции эмоций: включение камеры и распознование эмоций на лицах на видео.
        Функция завершается при нажатии на Escape

        :argument
        p_cam_num - номер камеры в системе
        p_freq_emo_class - частота (количество кадров) определения эмоции
        p_frame_color - цвет рамки выделения лица (RGB)
        p_label_color - цвет подписи эмоции (RGB)
        p_font_scale - размер (масштаб) подписи эмоции

        '''

        try:
            # Шрифт подписи эмоции
            font = cv2.FONT_HERSHEY_TRIPLEX

            # захватываем изображение с веб камеры
            cap = cv2.VideoCapture(p_cam_num)

            cnt = 0
            amo_label = ''
            while True:
                # получаем картинку с камеры
                _, img = cap.read()
                # конвертируем изображение в черно-белые (серые) цвета
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Детектируем лица
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                # Отрисовка рамок вокруг каждого лица
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), p_frame_color, 2)

                    # вырезаем лицо по рамке
                    crop_img = img[y:y + h, x:x + w]

                    if cnt == p_freq_emo_class:
                        # Подготавливаем изображение для отправки в модель
                        preprocessed_img = cv2.resize(crop_img,
                                                      dsize=(self.emo_classifier.image_size, self.emo_classifier.image_size),
                                                      interpolation=cv2.INTER_CUBIC)

                        preprocessed_img = np.asarray(preprocessed_img)
                        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

                        # получим предсказания
                        amo_label = self.emo_classifier.predict(self.emo_classifier.preprocess_input_func(preprocessed_img))

                    # if

                    cv2.putText(img, amo_label, (x - 3, y - 2), font, p_font_scale, p_label_color)

                # for

                if cnt == p_freq_emo_class:
                    cnt = 0
                else:
                    cnt = cnt + 1

                # вывод изображения
                cv2.imshow('img', img)
                # Заканчиваем, если была нажата кнопка "Escape"
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

            # while
        # try
        finally:
            cap.release()
            cv2.destroyAllWindows()
        # finally

    #run

#EmoDetector


if __name__ == '__main__':
    ed = EmoDetector()
    ed.run(0)