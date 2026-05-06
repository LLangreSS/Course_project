import json
import csv
import time
import requests

API_URL = "http://127.0.0.1:8000/verify/"
JSON_FILE = "test_cases.json"
CSV_FILE = "test_report.csv"
THRESHOLD = 0.85


def load_tests():
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_tests():
    tests = load_tests()
    report = []

    print(f"Начало прогона {len(tests)} тестов...\n" + "-" * 50)

    passed_count = 0

    for test in tests:
        start_time = time.time()

        # Формируем тело запроса согласно вашей схеме VerifyRequest
        payload = {
            "text": test["text"],
            "threshold": THRESHOLD
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            actual_status = data.get("status")
            details = data.get("results", [])

            # Извлекаем максимальную уверенность из фактов
            confidences = [
                fact.get("best_match", {}).get("similarity", 0.0)
                for fact in details if fact.get("best_match")
            ]
            max_conf = max(confidences) if confidences else 0.0

            error_msg = ""
        except Exception as e:
            actual_status = "ERROR"
            max_conf = 0.0
            error_msg = str(e)

        exec_time = round(time.time() - start_time, 2)

        # Проверка результата
        is_success = (actual_status == test["expected_status"])
        if is_success:
            passed_count += 1
            test_status = "ПРОЙДЕН"
        else:
            test_status = "ПРОВАЛЕН"

        comment = f"Время: {exec_time}с. Уверенность (поиск): {round(max_conf, 2)}."
        if not is_success:
            if error_msg:
                comment += f" Ошибка API: {error_msg}"
            else:
                comment += f" Ожидали {test['expected_status']}, получили {actual_status}."

        print(f"[{test_status}] ID: {test['id']} | {exec_time}s | {test['text'][:40]}...")

        report.append({
            "ID": test["id"],
            "Группа": test["group"],
            "Текст запроса": test["text"],
            "Ожидаемый статус": test["expected_status"],
            "Фактический статус": actual_status,
            "Результат": test_status,
            "Комментарий": comment
        })

    with open(CSV_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys(), delimiter=';')
        writer.writeheader()
        writer.writerows(report)

    print("-" * 50)
    print(f"Тестирование завершено. Успешно: {passed_count}/{len(tests)}")
    print(f"Отчет сохранен в файл: {CSV_FILE}")


if __name__ == "__main__":
    run_tests()