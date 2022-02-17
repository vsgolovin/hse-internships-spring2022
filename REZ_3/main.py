from collections import namedtuple
from typing import List, Callable
from copy import copy
import numpy as np
import cv2 as cv


INPUT_IMG = 'source.jpeg'       # исходное изображение
PROCESSED_IMG = 'interim.jpeg'  # обработанное изображение
OUTPUT_IMG = 'output.jpeg'      # исх. изображение с выделеными абзацами
MIN_HEIGHT = 5                  # минимальная высота строчки (пиксели)
MIN_FIRST_LINE_OFFSET = 30      # минимальный отступ первой строки
MIN_TITLE_SPACING = 20          # минимальный отступ после заголовка
LINE_THICKNESS = 2              # толщина линии выделяющей рамки
LINE_COLOR = (0, 0, 255)        # цвет этой линии (BGR)

# тип для выделения блока текста
Rectangle = namedtuple('Rectangle', list('xywh'))


def main():
    # чтение и обработка изображения для поиска контуров строк / слов
    img = cv.imread(INPUT_IMG)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray, (11, 3))
    _, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY_INV)
    kernel = np.ones((1, 31), dtype='uint8')
    dilate = cv.dilate(thresh, kernel, iterations=1)

    # сохранение промежуточного изображения
    cv.imwrite(PROCESSED_IMG, dilate)

    # поиск контуров и рамок (блоков)
    contours = cv.findContours(dilate, cv.RETR_EXTERNAL,
                               cv.CHAIN_APPROX_SIMPLE)[0]
    assert len(contours) > 0, 'No contours found'
    rectangles = []
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if h >= MIN_HEIGHT:
            rectangles.append(Rectangle(x, y, w, h))
    # сортировка по верхней границе
    rectangles = sorted(rectangles, key=lambda r: r.y)

    # объединение блоков в строки
    rectangles = group_rectangles(rectangles, same_line)
    # объёдинение блоков в абзацы
    rectangles = group_rectangles(rectangles, same_paragraph)

    # добавление рамок на исходное изображение + экспорт
    for r in rectangles:
        cv.rectangle(img, (r.x, r.y), (r.x + r.w, r.y + r.h),
                     LINE_COLOR, LINE_THICKNESS)
    cv.imwrite(OUTPUT_IMG, img)


def same_line(r1: Rectangle, r2: Rectangle) -> bool:
    """
    Проверяет, принадлежат ли два блока одной строке.
    """
    return min(r1.y + r1.h, r2.y + r2.h) >= max(r1.y, r2.y)


def same_paragraph(r1: Rectangle, r2: Rectangle) -> bool:
    """
    Проверяет, принадлежат ли два блока одному абзацу.
    """
    assert r2.y > r1.y
    if r2.y - (r1.y + r1.h) > MIN_TITLE_SPACING:
        return False
    return r2.x - r1.x < MIN_FIRST_LINE_OFFSET


def group_rectangles(rectangles: List[Rectangle],
                     to_join: Callable) -> List[Rectangle]:
    """
    Сгруппировать блоки в соответствии с условием.
    """
    assert len(rectangles) > 0
    r = copy(rectangles)  # не изменять исходный список
    prev = r[0]
    output = []
    for i in range(1, len(r)):
        if to_join(prev, r[i]):
            prev = join(prev, r[i])
        else:
            output.append(prev)
            prev = r[i]
    output.append(prev)
    return output


def join(r1: Rectangle, r2: Rectangle):
    """
    Объединить два блока.
    """
    x = min(r1.x, r2.x)
    y = min(r1.y, r2.y)
    w = max(r1.x + r1.w, r2.x + r2.w) - x
    h = max(r1.y + r1.h, r2.y + r2.h) - y
    return Rectangle(x, y, w, h)


if __name__ == '__main__':
    main()
