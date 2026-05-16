#!/usr/bin/env python
"""
manage.py — Утилита командной строки Django.

Стандартный скрипт Django для запуска административных задач:
    python manage.py runserver        — запуск dev-сервера
    python manage.py migrate          — применение миграций БД
    python manage.py collectstatic    — сбор статических файлов
    python manage.py createsuperuser  — создание администратора
"""
import os
import sys


def main():
    """Запускает административные задачи Django."""
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE",
        "project_config.settings"
    )
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Не удаётся импортировать Django. Убедитесь, что он установлен "
            "и доступен в переменной окружения PYTHONPATH. Возможно, "
            "вы забыли активировать виртуальное окружение?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
