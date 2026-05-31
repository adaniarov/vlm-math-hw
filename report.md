# Report

## Track

Выбранный трек:

```text
A
```

## Что реализовано

- [x] dataset.py
- [x] processor.py
- [x] model.py
- [x] train.py
- [x] benchmark.py

## Конфигурация

```text
config path: configs/track_a_cpu.yaml, configs/inference_math.yaml --toy
seed: 42
device: cpu
dtype: float32
max_steps: 3 
batch size: local 1, global 1 
```

## Результаты

```text
public tests: 14 passed (pytest -q tests_public)
train loss: smoke 3 шага 7.25 -> 8.03 -> 7.72; с --fast-train (2 шага) 8.03; loss конечный
benchmark accuracy: 0 
```

## Использованные ресурсы

```text
CPU/GPU: CPU Apple Silicon
VRAM: нет
время обучения: smoke ~1 c, benchmark ~1 c, public tests ~1 c
```

## Анализ ошибок

Для трека A vision encoder и LLM заменены на случайно инициализированные backbone (`tiny/local-or-mocked`), обучается только adapter. Это smoke режим поэтому accuracy нулевая: модель не выдает распознаваемую букву

Все примеры выдаются ноль, потому что ничего не предсказывается. 
1. toy_dev_000: катеты 5 и 12, нужна гипотенуза, верный ответ 13
2. toy_dev_001: прямоугольник 5x2, нужна площадь, верный ответ 10
3. toy_dev_003: столбчатая диаграмма, значение столбца Q, верный ответ 4

## Комментарии

Больше всего времени ушло на то, чтобы во всех частях пайплайна сходилось число visual-токенов. Логика такая: в промпт я ставлю ровно столько меток `image`, сколько эмбеддингов дает картинка (`num_tiles * num_image_tokens`). adapter  сжимает произвольное число патчей от энкодера до этого же числа, а при сборке входа vis эмбеддинги подставляются точно на позиции этих меток. Если число расходится хотя бы на один — merge сразу падает, так что именно тут пришлось быть внимательнее всего.

Что улучшил бы: подключить реальные ViT, instruct-LLM на GPU в треке B, добавить LoRA, обучить на math_vqa_medium.

## Критерии оценивания

См. файл [`GRADING.md`](GRADING.md).
