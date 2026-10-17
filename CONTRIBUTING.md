# Как мы работаем (шпаргалка для A и B)

Главное правило: **`main` всегда рабочий**. В `main` попадает только то, что прошло Pull Request
и одобрено напарницей. Прямой push в `main` запрещён настройками репозитория.

## 1. Один раз: подготовка

1. Аккаунт на GitHub + принять приглашение в репозиторий (придёт на почту).
2. В Colab: *File → Open notebook → вкладка GitHub* → вставить ссылку на репозиторий → видны все ноутбуки
   (выбрать нужную ветку в выпадающем списке). Новый ноутбук начинаем с копии `notebooks/_template.ipynb`.
3. (Если работаешь и локально) `git clone https://github.com/bkadyrbay-hash/lab1-supersense-llm.git`

## 2. Цикл работы над одним этапом (например, A4)

```
1. Создать ветку        →  2. Работать в Colab     →  3. Сохранить в ветку
4. Открыть Pull Request →  5. Напарница смотрит    →  6. Merge в main
```

**1. Ветка.** На странице репо: кнопка с названием ветки (`main ▾`) → вписать `a/A4-icl-baseline`
(или `b/B2-lora-baseline`) → *Create branch from main*. Одна ветка = один этап.

**2. Colab.** Открыть ноутбук из GitHub (из **своей** ветки). Первая ячейка сама клонирует репо
и подтягивает `src/`. Если нужен ещё не смёрженный код из своей ветки — в первой ячейке `BRANCH = "a/A4-…"`.

**3. Сохранить ноутбук в GitHub.**
   - *Edit → Clear all outputs* (чтобы diff был читаемым и файл маленьким);
   - *File → Save a copy in GitHub* → репозиторий, **своя ветка**, путь `notebooks/01_icl.ipynb`,
     сообщение: `[A4] zero-shot ICL baseline on rand-dev`. Галочку «Include a link to Colab» снять.

**Файлы `.py`, `.csv`, `.png`, `.md`** из Colab так не сохранить. Варианты:
   - на GitHub нажать клавишу `.` → откроется веб-VS Code (github.dev) → правка/загрузка файлов
     (перетащить CSV/PNG, скачанные из Colab) → *Source Control* → commit в свою ветку;
   - или в репо *Add file → Upload files* → выбрать свою ветку;
   - или локально через `git` (см. ниже).

**4. Pull Request.** GitHub сам предложит *Compare & pull request*. Заполнить шаблон
(что сделано, какие строки добавлены в `metrics.csv`). Reviewer — напарница.

**5. Ревью** (10 минут, не больше): прочитать код, задать вопрос в комментарии, если строка непонятна —
так обе сможем объяснить весь код преподавателю. *Approve*.

**6. Merge** — кнопка *Squash and merge*. Ветку удалить (GitHub предлагает).

## 3. Правила, которые экономят нервы

- **Чужой файл не редактируем** (владельцы — в таблице ниже). Нужна правка — пишем напарнице
  или открываем PR и отмечаем её.

  | A (ICL) | B (LoRA) | обе |
  |---|---|---|
  | `README.md`, `requirements.txt`, `.gitignore` | `src/evaluate.py` | `results/metrics.csv` (только дописываем) |
  | `src/data.py`, `src/prompts.py`, `src/label_scoring.py` | `notebooks/02_lora.ipynb`, `03_camembert.ipynb` | `results/predictions/`, `results/figures/` |
  | `notebooks/00_data_exploration.ipynb`, `01_icl.ipynb` | `notebooks/04_error_analysis.ipynb` (каркас) | анализ в `04_error_analysis` |
  | `docs/model_choice.md`, `docs/prompt_literature.md` | `docs/lora_notes.md` | |
- **`results/metrics.csv` — только дописываем строки** через `log_experiment()`, ничего не удаляем.
  Git настроен (`.gitattributes`, `merge=union`) так, что строки из двух веток склеиваются без конфликта.
- **Имена файлов предсказаний уникальные**: `{method}__{config_id}__{split}.csv` — конфликтов не бывает.
- Датасет, веса и LoRA-адаптеры **не коммитим** (`.gitignore`); адаптеры — на Google Drive.
- `freq-test` / `rand-test` — только на этапах A8 / B7, один раз.
- Перед PR проверка `ruff` запускается автоматически (вкладка *Actions*): ловит неиспользуемые импорты
  и переменные — преподаватель обещал снимать за «useless instructions».

## 4. Локально через git (по желанию)

```bash
git switch main && git pull                 # свежий main
git switch -c a/A6-label-scoring            # новая ветка
# ... правки ...
git add src/label_scoring.py
git commit -m "[A6] log-prob scoring of single-token labels"
git push -u origin a/A6-label-scoring       # затем PR на сайте
```

Если ветка отстала от `main` (напарница уже смёржила что-то): на странице PR кнопка *Update branch*.

## 5. Сообщения коммитов

`[ID этапа] что сделано, в повелительном наклонении`, на английском:
`[A5] add 5/25/50-shot runs on rand-dev`, `[B3] compare q,v vs all-linear LoRA targets`,
`[shared] fix macro-F1 on missing labels`.
