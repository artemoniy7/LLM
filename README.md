# RP-LLM — C++17 mini Transformer

Учебная LLM для RP и диалогов, полностью на C++17 без Python runtime и внешних ML-библиотек.

## Что внутри
- decoder-only Transformer;
- causal self-attention;
- token + positional embeddings;
- LayerNorm, GELU, residual connections;
- собственный reverse-mode autodiff для матриц;
- Cross-Entropy loss;
- AdamW + gradient clipping;
- бинарные checkpoints;
- byte-level UTF-8 tokenizer (256 byte tokens + специальные токены);
- train/generate CLI;
- пример RP-датасета.

## Ограничения
Это намеренно маленькая CPU-ориентированная модель. Она предназначена для обучения архитектуре и экспериментов, а не для качества уровня современных LLM. Текущая конфигурация: 3 слоя, d=128, FFN=384, context=128, vocab=260.

## Сборка
```bash
cmake -S . -B build
cmake --build build --config Release
```

## Обучение
```bash
./build/rp_llm train data/rp.txt 1000 model.bin
```
На Windows путь к exe обычно `build/Release/rp_llm.exe`.

## Генерация
```bash
./build/rp_llm generate model.bin "SYSTEM: Отвечай как персонаж. USER: *входит в таверну* Добрый вечер. ASSISTANT:"
```

Для реального качества RP следующий этап — BPE/SentencePiece-подобный tokenizer, packed dataset, более длинный context, multi-head attention, batching/gradient accumulation и BF16/AVX2/oneDNN backend. На i7-12700H лучше сначала проверить корректность обучения на этой маленькой конфигурации.
