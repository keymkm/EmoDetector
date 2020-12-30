'''В данном скрипте описаны вспомогательные функции для проекта классификации эмоций'''

from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#----------------------------------

def get_paths (p_root_dir = ''):
    '''
    Получение путей до основных директорий и файлов проекта
    :argument
    p_root_dir - корневая директория проекта. Если проект выполняется локально, то лучше оставить все пути относительными,
      тогда можно оставить корневую папку пустой. Если проект выполняется на google drive, то указываются абсолутные пути
      на google drive

    :return
    paths - dict {} со следующей структурой:
      data_dir - путь до директории с данными
      train_data_dir - путь до деритории с тренировочными данными
      test_data_dir - путь до дериктории с тестовыми данными
      class_names_path - путь до файла с названиями классов эфоций. Названия классов перечислены в порядке выдачи модели
      model_dir - путь до директории с файлами модели
      best_checkpoint_path - путь до файла модели с лучшими результатми обучения в формате h5
    '''

    paths = {
             'data_dir': 'data',
             'model_dir': 'model'
             }

    if p_root_dir != '':
        paths['data_dir'] = p_root_dir + '/' + paths['data_dir']
        paths['model_dir'] = p_root_dir + '/' + paths['model_dir']
    # if

    assert Path(paths['data_dir']).exists(), 'Не найдена директория с данными "{}"'.format(paths['data_dir'])
    paths['train_data_dir'] = paths['data_dir'] + '/train'
    paths['train_aug_data_dir'] = paths['data_dir'] + '/train_aug'
    paths['val_data_dir'] = paths['data_dir'] + '/val'
    paths['test_data_dir'] = paths['data_dir'] + '/test_kaggle'
    paths['class_names_path'] = paths['data_dir'] + '/class_names.txt'

    Path(paths['model_dir']).mkdir(exist_ok=True, parents=True)
    paths['best_checkpoint_path'] = paths['model_dir'] + '/emo_classificator_best'

    return paths
#get_paths

#----------------------------------
def preprocess_image(p_img_path, p_img_target_size, preprocess_func):
    '''
    Преобразование картинки для подачи в модель
    :argument
    p_img_path - путь до картинки
    p_img_target_size - целевой размер картинки tuple (row_cnt, col_cnt)
    preprocess_func - функция специфичная для преобразования картинки для конкретной модели
    '''

    img = image.load_img(p_img_path, target_size=p_img_target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    return preprocess_func(img)

# preprocess_image

#----------------------------------
def show_img_emo(p_img_paths, p_img_emo_idxs, p_img_emo_pred_idxs, p_emo_classes_names):
    '''
    Отображение картинок с реальной и предсказанной эмоцией

    :argument
    p_img_paths - пути до картинки
    p_img_emo_idxs - список индексов реальных эмоций
    p_img_emo_pred_idxs - список индексов реальных эмоций
    p_emo_classes_names - список классов эмоций
    '''

    # расчитаем размерность сетки для вывода результатов
    img_cnt = len(p_img_paths)

    col_cnt = 4
    row_cnt = 1

    if col_cnt >= img_cnt:
        col_cnt = img_cnt
    else:
        row_cnt = img_cnt // col_cnt

    fig = plt.figure(tight_layout=True)

    # задаем размерность сетки для размещения картинок
    gs = fig.add_gridspec(row_cnt, col_cnt)
    fig.set_size_inches(10, 10)

    row = 0
    col = 0
    for i in range(img_cnt):
        ax = fig.add_subplot(gs[row, col])

        if col < col_cnt - 1:
            col = col + 1
        else:
            col = 0
            row = row + 1
        # else

        imgplot = plt.imshow(mpimg.imread(p_img_paths[i]))

        color = 'green'
        if p_img_emo_idxs[i] != p_img_emo_pred_idxs[i]:
            color = 'red'

        ax.set_title('класс (реальн): {}\nкласс (предск): {}'.format(
            p_emo_classes_names[p_img_emo_idxs[i]],
            p_emo_classes_names[p_img_emo_pred_idxs[i]]), color=color)

    # for

    plt.show()


# show_img_emo

#----------------------------------
def test_model_prediction(p_model, p_img_paths, p_img_emo_idxs, p_img_target_size, preprocess_func, p_emo_classes_names):
    '''
    Демонстрация предсказания модели на тестовом наборе картинок

    :argument
    p_model - модель для классификации эмоций
    p_img_paths - пути до картинки
    p_img_emo_idxs - список индексов реальных эмоций
    p_img_target_size - целевой размер картинки tuple (row_cnt, col_cnt)
    p_emo_classes_names - список классов эмоций
    preprocess_func - функция специфичная для преобразования картинки для конкретной модели

    '''
    img_cnt = len(p_img_paths)

    # соберем набор изображений для предсказания
    preprocessed_imgs = None
    for i in range(img_cnt):
        if i == 0:
            preprocessed_imgs = preprocess_image(p_img_paths[i], p_img_target_size, preprocess_func)
        else:
            preprocessed_imgs = np.append(preprocessed_imgs,
                                          preprocess_image(p_img_paths[i], p_img_target_size, preprocess_func),
                                          axis=0)

    # for

    # получим предсказания
    predictions = p_model.predict(preprocessed_imgs)
    predicted_img_labels = [np.argmax(predictions[i]) for i in range(predictions.shape[0])]

    show_img_emo(p_img_paths, p_img_emo_idxs, predicted_img_labels, p_emo_classes_names)

# test_model_prediction

#----------------------------------
#!!!Проверка функций
#print(get_paths ())