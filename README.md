# mipt2024f_dankov_n_s
Точная локализация бар-кодов



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

# Bounding box

Тут уже интереснее. Есть интересная [статья](https://web.archive.org/web/20140602071142id_/http://www.fit.vutbr.cz:80/~herout/papers/2012-SCCG-QRtiles.pdf) про локализацию qr-кодов с помощью преобразования Хафа. Так же какая-то информация имеется на [статье](https://smartengines.medium.com/qr-code-localization-the-important-recognition-step-that-has-been-neglected-8b3ed4e8037) от самих Smart Engines.

Но общий подход предполагается с использованием нейросетки YOLO, так как она хорошо справляется с задачей детекции объектов как на статических картинках, так и на видеороликах. Вот [пример](https://www.dynamsoft.com/codepool/qr-code-detect-decode-yolo-opencv.html) имплементации.

## 01.10.2024
IoU плохо, так как есть union, а нам надо чтобы весь QR попал в BB.

Как нужно обходить QR при точной локализации

Порядок обхода.  Против часовой.

Эталонная ориентация. Левый нижний угол эталонного бар кода.

mipt2024f_fam_i_o

mipt2024f_recsys

vgg group VIA