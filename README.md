# LLM

Минимальный учебный LLM-проект на C++.

## Основные файлы

- `config.json` — единая настройка датасета, модели, обучения, checkpoint и чата.
- `train.cpp` — обучение модели по данным из `config.json`.
- `chat.cpp` — чат с моделью из checkpoint.
- `include/` — заголовки ядра модели, токенизатора, конфигурации и JSON-библиотеки.
- `src/` — реализация tensor/layer/transformer.
- `data/` — датасеты.

## Датасет

По умолчанию используется PIPPA JSONL:

```json
"dataset_path": "data/pippa.jsonl",
"dataset_format": "pippa_jsonl"
```

Если нужен старый формат `user|assistant`, поставь:

```json
"dataset_path": "data/dialogues.txt",
"dataset_format": "pipe"
```

## Компиляция

Для обычной сборки:

```bash
g++ -std=c++17 -Iinclude train.cpp src/tensor.cpp src/layer.cpp src/transformer.cpp -o train
g++ -std=c++17 -Iinclude chat.cpp src/tensor.cpp src/layer.cpp src/transformer.cpp -o chat
```

Для более быстрого обучения лучше включить оптимизации:

```bash
g++ -O3 -DNDEBUG -march=native -std=c++17 -Iinclude train.cpp src/tensor.cpp src/layer.cpp src/transformer.cpp -o train
```

## Запуск

```bash
./train
./chat
```

Обучение автоматически загружает checkpoint из `checkpoint_path`, если он есть, и сохраняет лучший/финальный checkpoint туда же. Чат использует тот же `config.json` и загружает тот же checkpoint.
