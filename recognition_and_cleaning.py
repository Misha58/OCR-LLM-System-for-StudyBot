from paddleocr import PaddleOCR
import os
import requests
import json
import pandas as pd
from datetime import datetime
import re
import glob

# Конфигурация Ollama — адрес локального сервера и название модели
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


class TextCleaner:
    """Класс для очистки OCR-текста с помощью языковой модели через Ollama."""

    def __init__(self, model_name=OLLAMA_MODEL):
        self.model_name = model_name
        self.url = OLLAMA_URL

    def clean_text(self, raw_text):
        """
        Отправляет сырой OCR-текст в Ollama и возвращает очищенный результат.
        Промпт инструктирует модель убрать мусор (метки вариантов, элементы интерфейса,
        OCR-артефакты) и сохранить вопрос и варианты ответов в строгом формате.
        """
        prompt = f"""Ты редактор. Перед тобой OCR-текст с мусором. Твоя задача — убрать мусор и вывести результат в нужном формате.

МУСОР (удалить безвозвратно):
- Номер вопроса: "Вопрос #1", "Вопрос #25" и т.п.
- Элементы интерфейса: "Удалить выбранные шаги", "Далее", "Назад", "Закрыть", "Портал", "Входящие"
- Служебные строки: "Укажите верную последовательность:", "Выберите вариант ответа:"
- Пустые ячейки и заглушки: строки только из букв без смысла ("А б В Г Д"), строки только из нулей ("0 0 0 0")
- Одиночные буквы-метки на отдельной строке: "А", "Б", "В", "а", "б" — это метки вариантов, не текст

СОХРАНИТЬ (дословно, ничего не менять):
- Полный текст вопроса
- Полный текст описания/условия задачи — каждое предложение, каждый абзац
- Каждый вариант ответа — включая "Затрудняюсь ответить", "Все вышеперечисленное"
- Код — полностью, с отступами

ВАРИАНТЫ С БУКВЕННЫМИ МЕТКАМИ:
Если в OCR вариант записан как одиночная буква на одной строке и текст на следующей — объедини их в один вариант, метку убери, текст сохрани.
Пример: "А" + "Запрошу информацию" → 1) Запрошу информацию

ИСПРАВИТЬ OCR-АРТЕФАКТЫ:
- Латиница, похожая на кириллицу: B→В, E→Е, F→Г, P→Р, C→С, O→О
- Символы-мусор: | ¦ █ □ → подходящая буква по контексту
- Двойные пробелы → одинарный

СТРОГИЕ ЗАПРЕТЫ:
- Не добавляй текст от себя
- Не сокращай и не перефразируй
- Не придумывай описание если его нет
- Не пропускай ни один вариант ответа
- Не дублируй варианты
- Код переписывай дословно, без изменений
- НЕ ФОРМАТИРУЙ И НЕ РЕДАКТИРУЙ ТЕКСТ

ПРИМЕР:
Входные данные:
Вопрос #25
Какие из предложенных действий вы выполните? Расположите их в правильной последовательности
Российский разработчик платформы для проведения дистанционных занятий физкультурой с автоматическим контролем выполненных упражнений и панелью администрирования для тренеров и преподавателей решил
усовершенствовать свой продукт. На платформе представлено 25 упражнений, на основе которых пользователи формируют комплексы упражнений с учетом индивидуальных физических особенностей занимающихся. В
личном кабинете преподавателя настроен автоматический контроль времени и скорости выполнения упражнений, а также ведется учет прогресса на временном промежутке. Руководство компании решило разработать
дополнительный модуль для лиц с ограниченными возможностями здоровья и для пожилых людей. В модуле предполагается реализовать возможность проведения дистанционных занятий с ними под контролем врачей-
реабилитологов. Вы приняты на работу в компанию для разработки данного модуля. Все ИТ-специалисты привлекаются на аутсорсинге, и Вам предстоит выстроить работу с ними как руководителю проекта, сформировать
ресурсный план и определить цели проекта.
А
Запрошу у руководителя компании информацию о бюджете проекта
б
Определю цели проекта и согласую их с руководителем компании
B
Предложу руководителю компании привлечь на аутсорсинге Ит-компанию, специализирующуюся на разработке подобных решений
F
Проведу оценку ресурсов по проекту: кадровые, технологические, финансовые, временные
д
Разработаю бюджет проекта и согласую его с руководителем компании
E
Обсужу с руководителем компании и другими заинтересованными сторонами (Ит-специалистами, пользователями платформы) риски по проекту и проведу оценку их влияния на проект
ж
Подготовлю служебную записку на имя руководителя компании с запросом технического задания по проекту
3
Изучу лучшую практику управления Ит-проектами, чтобы внедрить в работу самые передовые методы
и
Найду в открытых источниках примеры по реализации подобных проектов и пообщаюсь с их разработчиками, чтобы взять за основу их опыт в управлении проектом
й
Выясню ожидания от проекта основных заинтересованных сторон: руководитель компании, потенциальные пользователи платформы: люди с ОВЗ, врачи-реабилитологи, ИТ-специалисты, которые участвовали в разработке
 платформы и будут участвовать в разработке модуля
Укажите верную последовательность:
Удалить выбранные шаги
А
б
B
F
Д
E
ж
з
И
й
о
о
о
о
о
о
о
о
о
о

Результат:
Вопрос:
Какие из предложенных действий вы выполните? Расположите их в правильной последовательности

Российский разработчик платформы для проведения дистанционных занятий физкультурой с автоматическим контролем выполненных упражнений и панелью администрирования для тренеров и преподавателей решил
усовершенствовать свой продукт. На платформе представлено 25 упражнений, на основе которых пользователи формируют комплексы упражнений с учетом индивидуальных физических особенностей занимающихся. В
личном кабинете преподавателя настроен автоматический контроль времени и скорости выполнения упражнений, а также ведется учет прогресса на временном промежутке. Руководство компании решило разработать
дополнительный модуль для лиц с ограниченными возможностями здоровья и для пожилых людей. В модуле предполагается реализовать возможность проведения дистанционных занятий с ними под контролем врачей-
реабилитологов. Вы приняты на работу в компанию для разработки данного модуля. Все ИТ-специалисты привлекаются на аутсорсинге, и Вам предстоит выстроить работу с ними как руководителю проекта, сформировать
ресурсный план и определить цели проекта.

Варианты:
1) Запрошу у руководителя компании информацию о бюджете проекта
2) Определю цели проекта и согласую их с руководителем компании
3) Предложу руководителю компании привлечь на аутсорсинге Ит-компанию, специализирующуюся на разработке подобных решений
4) Проведу оценку ресурсов по проекту: кадровые, технологические, финансовые, временные
5) Разработаю бюджет проекта и согласую его с руководителем компании
6) Обсужу с руководителем компании и другими заинтересованными сторонами (Ит-специалистами, пользователями платформы) риски по проекту и проведу оценку их влияния на проект
7) Подготовлю служебную записку на имя руководителя компании с запросом технического задания по проекту
8) Изучу лучшую практику управления Ит-проектами, чтобы внедрить в работу самые передовые методы
9) Найду в открытых источниках примеры по реализации подобных проектов и пообщаюсь с их разработчиками, чтобы взять за основу их опыт в управлении проектом
10) Выясню ожидания от проекта основных заинтересованных сторон: руководитель компании, потенциальные пользователи платформы: люди с ОВЗ, врачи-реабилитологи, ИТ-специалисты, которые участвовали в разработке платформы и будут участвовать в разработке модуля

ФОРМАТ ВЫВОДА (строго, ничего лишнего до и после):
Вопрос:
[полный текст вопроса + полный текст описания если есть]

Варианты:
1) [текст]
2) [текст]
...

OCR-текст:
{raw_text}"""

        try:
            # Отправляем POST-запрос к локальному серверу Ollama
            response = requests.post(
                self.url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    # Низкая температура для детерминированного вывода
                    "options": {"temperature": 0.1}
                },
                timeout=250
            )
            if response.status_code == 200:
                # Извлекаем поле "response" из JSON-ответа Ollama
                return response.json().get("response", "").strip()
            else:
                print(f"  Ошибка Ollama: HTTP {response.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            print("  Не удалось подключиться к Ollama. Убедитесь, что сервер запущен: ollama serve")
            return None
        except requests.exceptions.Timeout:
            print("  Таймаут запроса к Ollama")
            return None
        except Exception as e:
            print(f"  Ошибка при обращении к Ollama: {e}")
            return None


def parse_cleaned_text(cleaned_text):
    """
    Парсит очищенный текст, возвращённый моделью, в структурированный словарь.
    Ожидает формат с секциями 'Вопрос:' и 'Варианты:'.
    Возвращает dict с ключами 'question' (строка) и 'options' (список строк).
    """
    question = ""
    options = []

    # Извлекаем блок вопроса — всё между "Вопрос:" и "Варианты:" (или концом строки)
    question_match = re.search(r"Вопрос:\s*\n(.*?)(?=\nВарианты:|\Z)", cleaned_text, re.DOTALL)
    if question_match:
        question = question_match.group(1).strip()

    # Извлекаем блок вариантов — всё после "Варианты:"
    options_match = re.search(r"Варианты:\s*\n(.*)", cleaned_text, re.DOTALL)
    if options_match:
        options_text = options_match.group(1).strip()
        # Находим строки вида "1) текст", "2) текст" и т.д.
        options = re.findall(r"^\d+\)\s*.+", options_text, re.MULTILINE)
        options = [o.strip() for o in options]

    return {"question": question, "options": options}


def extract_text_from_ocr_json(json_data):
    """
    Извлекает весь распознанный текст из JSON-файла, сохранённого PaddleOCR.
    Поддерживает два возможных формата ключей: 'rec_texts' и 'texts'.
    Возвращает единую строку с текстом, где каждый фрагмент на новой строке.
    """
    full_text = ""
    # Формат PaddleOCR v3+: ключ 'rec_texts'
    if 'rec_texts' in json_data:
        for text in json_data['rec_texts']:
            if text and text.strip():
                full_text += text + "\n"
    # Альтернативный формат: ключ 'texts'
    if 'texts' in json_data:
        for text in json_data['texts']:
            if text and text.strip():
                full_text += text + "\n"
    return full_text.strip()


def process_single_ocr_file(json_path, text_cleaner, output_folder, all_questions_data):
    """
    Обрабатывает один JSON-файл с результатами OCR:
      1. Читает JSON и извлекает сырой текст.
      2. Сохраняет сырой текст в подпапку raw_texts.
      3. Отправляет текст на очистку в Ollama через TextCleaner.
      4. Парсит очищенный ответ в структурированный формат.
      5. Сохраняет очищенный текст в подпапку cleaned_texts.
      6. Добавляет строку с вопросом и вариантами в общий список all_questions_data.
    Возвращает True при успехе, False при ошибке.
    """
    try:
        print(f"Обработка файла: {os.path.basename(json_path)}")

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Извлекаем текст из JSON
        full_text = extract_text_from_ocr_json(json_data)
        if not full_text:
            print("  Не удалось извлечь текст из JSON")
            return False

        print(f"  Извлечено символов: {len(full_text)}")

        # Сохраняем сырой (неочищенный) текст для отладки и проверки
        raw_folder = os.path.join(output_folder, "raw_texts")
        os.makedirs(raw_folder, exist_ok=True)
        raw_filename = os.path.splitext(os.path.basename(json_path))[0] + "_raw.txt"
        with open(os.path.join(raw_folder, raw_filename), 'w', encoding='utf-8') as f:
            f.write(full_text)

        # Отправляем текст в Ollama для очистки от мусора
        cleaned_text = text_cleaner.clean_text(full_text)
        if not cleaned_text:
            print("  LLM не вернул результат, пропускаем файл")
            return False

        # Парсим ответ модели в структуру {question, options}
        parsed = parse_cleaned_text(cleaned_text)

        # Сохраняем очищенный текст
        cleaned_folder = os.path.join(output_folder, "cleaned_texts")
        os.makedirs(cleaned_folder, exist_ok=True)
        cleaned_filename = os.path.splitext(os.path.basename(json_path))[0] + "_cleaned.txt"
        with open(os.path.join(cleaned_folder, cleaned_filename), 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        print(f"  Сохранён: {cleaned_filename} | Вариантов: {len(parsed['options'])}")

        # Формируем строку для DataFrame с динамическими колонками вариантов (option_1, option_2, ...)
        row = {
            'source_file': os.path.basename(json_path),
            'question': parsed.get('question', ''),
        }
        for i, option in enumerate(parsed['options'], start=1):
            row[f'option_{i}'] = option

        all_questions_data.append(row)
        return True

    except Exception as e:
        print(f"  Ошибка при обработке {json_path}: {e}")
        return False


def run_ocr_on_images(input_folder, output_folder):
    """
    Запускает PaddleOCR на всех изображениях в input_folder (рекурсивно).
    Для каждого изображения сохраняет JSON с результатами OCR в output_folder.
    Имена JSON-файлов формируются из относительных путей изображений,
    чтобы избежать конфликтов при одинаковых именах файлов в разных подпапках.
    Возвращает список путей к сгенерированным JSON-файлам.
    """
    # Инициализируем OCR-движок для русского языка без лишних модулей
    ocr = PaddleOCR(
        lang="ru",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    # Рекурсивно собираем все пути к изображениям
    image_paths = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_paths.append(os.path.join(root, file))

    print(f"Найдено изображений: {len(image_paths)}")

    json_files = []
    for idx, image_path in enumerate(image_paths, 1):
        image_file = os.path.basename(image_path)
        print(f"\n[{idx}/{len(image_paths)}] OCR: {image_path}")
        try:
            result = ocr.predict(image_path)
            result[0].save_to_json(output_folder)
            # Строим уникальное имя JSON из относительного пути изображения
            rel_path = os.path.relpath(image_path, input_folder)
            safe_name = rel_path.replace(os.sep, '_')
            json_filename = os.path.splitext(safe_name)[0] + "_res.json"
            json_path = os.path.join(output_folder, json_filename)
            json_files.append(json_path)
        except Exception as e:
            print(f"  Ошибка при OCR {image_file}: {e}")

    return json_files


def main():
    """
    Главная функция пайплайна:
      1. Ищет уже существующие JSON-файлы с результатами OCR.
      2. Если JSON не найдены — запускает OCR на изображениях.
      3. Для каждого JSON вызывает обработку: извлечение, очистку, парсинг.
      4. Сохраняет все вопросы и варианты ответов в Excel-файл с автоподбором ширины колонок.
    """
    # Инициализируем очиститель текста с настройками по умолчанию
    text_cleaner = TextCleaner()

    input_folder = r"C:\Users\Redmi\Desktop\скриншоты"
    output_folder = r"C:\Users\Redmi\Documents\project\ТЕСТ_Скрины\result3"
    os.makedirs(output_folder, exist_ok=True)

    # Список словарей для построения итогового DataFrame
    all_questions_data = []

    # Ищем уже готовые JSON с результатами OCR в папке с исходниками
    json_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.json') and '_res' in file.lower():
                json_files.append(os.path.join(root, file))

    # Также проверяем папку с результатами (на случай предыдущего запуска)
    json_files.extend(glob.glob(os.path.join(output_folder, "*_res.json")))
    json_files = list(set(json_files))  # убираем дубликаты

    if not json_files:
        print("JSON файлов не найдено. Запускаем OCR...")
        json_files = run_ocr_on_images(input_folder, output_folder)

    print(f"\nФайлов для обработки: {len(json_files)}")

    # Последовательно обрабатываем каждый JSON-файл
    for idx, json_file in enumerate(json_files, 1):
        print(f"\n[{idx}/{len(json_files)}] ", end="")
        process_single_ocr_file(json_file, text_cleaner, output_folder, all_questions_data)
        print("-" * 60)

    # Сохраняем итоговые данные в Excel
    if all_questions_data:
        # DataFrame автоматически заполнит NaN там, где вариантов меньше максимума
        df = pd.DataFrame(all_questions_data).fillna('')

        # Сортируем колонки: source_file, question, option_1, option_2, ...
        option_cols = sorted(
            [c for c in df.columns if c.startswith('option_')],
            key=lambda x: int(x.split('_')[1])
        )
        df = df[['source_file', 'question'] + option_cols]

        # Формируем имя файла с временной меткой во избежание перезаписи
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_folder = os.path.join(output_folder, "excel_reports")
        os.makedirs(excel_folder, exist_ok=True)
        excel_path = os.path.join(excel_folder, f"questions_{timestamp}.xlsx")

        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Questions', index=False)
            worksheet = writer.sheets['Questions']
            # Автоподбор ширины колонок по максимальной длине содержимого (не более 80 символов)
            for column in worksheet.columns:
                max_length = max((len(str(cell.value or '')) for cell in column), default=0)
                col_letter = column[0].column_letter
                worksheet.column_dimensions[col_letter].width = min(max_length + 2, 80)

        print(f"\nExcel сохранён: {excel_path}")
        print(f"Всего вопросов: {len(all_questions_data)}")
        print(f"Максимум вариантов в одном вопросе: {len(option_cols)}")
    else:
        print("\nНет данных для сохранения")

    print("\n" + "=" * 60)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print(f"Результаты: {output_folder}")


if __name__ == "__main__":
    main()