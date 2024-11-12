# Точная локализация бар-кода
Точная локализация бар-кодов. Необходимо проработать алгоритм, который выделяет на изображении всех бар-коды, при этом на изображении он должен быть только один (то есть после грубой локализации баркодов).


# API
- Входные данные: Карта Instance segmentation. То есть, отображение вида ``R x R -> N + {0}`` в виде массива.
- Выходные данные: Json объект, вида:
```json
[
    {
        "id": uint,
        "x": list[uint],
        "y": list[uint],
    }
]
```
Причем, точки в массиве расположены таким образом, что при обходе контура слева будет расположен сам баркод, причем первой точкой всегда будет являться левая-верхняя точка баркода, которая была бы у шаблонного.

# Точная детекция

После сегментации, используя функцию `aproxPoly()` из OpenCV, можно получить примерный обрамляющий штрихкод многоугольник. То есть хотим выбрать основное ребро бар-кода, которое будем считать базой.

А потом можно делать что-то подобное.

В общем мега имбовая стать на эту тему [Brylka, R., Schwanecke, U., & Bierwirth, B. (2020). Camera Based Barcode Localization and Decoding in Real-World Applications. 2020 International Conference on Omni-Layer Intelligent Systems (COINS)](https://sci-hub.ru/https://ieeexplore.ieee.org/abstract/document/9191416).

# Текущие результаты
Прикольная [статья](https://pyimagesearch.com/2014/11/24/detecting-barcodes-images-python-opencv/) про имплементацию алгоритма, основанном на примении различных фильтров с помощью сверток. На текущий момент, алгоритм либо хорошо выделяет только вертикальные штрихкоды, либо только горизонтальные.

![](images/output.png)
![](images/2.png)

## 01.10.2024
IoU плохо, так как ес

Как нужно обходить QR при точной локализации

Порядок обхода.  Против часовой.

Эталонная ориентация. Левый нижний угол эталонного бар кода.

mipt2024f_fam_i_o

mipt2024f_recsys

vgg group VIA