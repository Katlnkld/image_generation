# pix2pix
## Image2Image generation with pix2pix

## Генерация изображений с помощью pix2pix

В первой части проекта была решена задача восстановления фасадов зданий по изображению его меток. Была написана модель pix2pix. Параметры этой модели были взяты из оригинальной статьи. Результаты работы модели на новых изображениях можно видеть ниже.
![alt text](images/facade1.JPG)
Видно, что модель правильно генерирует окна, балконы, двери и другие элементы фасадов. Однако, при генерации заднего фона и крыши у модели возникают некоторые проблемы. Дообучение не помогло, так как модель переобучается, возможно поможет экспериментирование с архитектурой и параметрами модели.


Во второй части необходимо было выбрать новую задачу и решить ее с помощью написанной модели. Я выбрала задачу улучшения качества и восстановления информации сильно затемненных фото. На картинках ниже представлены результаты работы модели.
![alt text](images/light2.JPG)
Модель обучилась хорошо и позволяет улучшать качество затемненных фотографий.

Больше примеров с результатами работы модели можно найти в папке */images*
