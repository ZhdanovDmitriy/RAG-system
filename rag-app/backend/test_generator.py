#!/usr/bin/env python3
"""
Скрипт для тестирования генератора вопросов
Использование: python test_generator.py
"""

import json
from pathlib import Path
from app.question_generator import QuestionGenerator

def test_generator():
    # Путь к текстовому файлу
    txt_path = Path("../sources/DM2024_module4.txt")
    
    if not txt_path.exists():
        print(f"Файл не найден: {txt_path}")
        print(f"Пытаемся найти в источниках...")
        txt_path = Path("sources/DM2024_module4.txt")
        
    if not txt_path.exists():
        print(f"Файл не найден: {txt_path}")
        import os
        print(f"Текущая директория: {os.getcwd()}")
        print(f"Содержимое текущей директории: {os.listdir('.')}")
        return
    
    # Читаем текст
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Загружено текста: {len(text)} символов")
    print()
    
    # Генерируем вопросы
    generator = QuestionGenerator(text)
    questions = generator.generate_all_questions()
    
    print(f"Сгенерировано вопросов: {len(questions)}")
    print()
    
    # Показываем примеры вопросов
    for i, q in enumerate(questions[:5]):
        print(f"--- Вопрос {i+1} ---")
        print(f"ID: {q['id']}")
        print(f"Вопрос: {q['question']}")
        print(f"Варианты ответов ({len(q['answers'])}):")
        for j, answer in enumerate(q['answers'][:3], 1):
            print(f"  {j}. {answer[:80]}...")
        print()
    
    # Сохраняем в JSON
    output_path = Path("../sources/test_questions_generated.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    
    print(f"Вопросы сохранены в: {output_path}")
    
    # Статистика
    total_answers = sum(len(q.get('answers', [])) for q in questions)
    print(f"\nСтатистика:")
    print(f"  Всего вопросов: {len(questions)}")
    print(f"  Всего ответов: {total_answers}")
    print(f"  Среднее ответов на вопрос: {total_answers / len(questions):.1f}")

if __name__ == "__main__":
    test_generator()
