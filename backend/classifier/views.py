"""
views.py — Представления страниц веб-интерфейса.

Содержит view-функции для отображения HTML-страниц приложения:
    • главная страница с формой загрузки/записи аудио;
    • страница "О проекте" с информацией о датасете и моделях.
"""
from django.shortcuts import render
from django.utils.translation import gettext_lazy as _


def home(request):
    """
    Главная страница: форма загрузки аудио и отображения результата.

    Шаблон: home.html
    """
    context = {
        "page_title": _("Классификация йеменских диалектов"),
    }
    return render(request, "home.html", context)


def about(request):
    """
    Страница "О проекте": краткое описание системы и датасета.

    Шаблон: about.html
    """
    context = {
        "page_title": _("О проекте"),
        # Данные о датасете для отображения
        "dataset_stats": {
            "total":      3315,
            "train":      2302,
            "validate":   758,
            "test":       255,
            "dialects":   11,
            "samples":    755,
        },
    }
    return render(request, "about.html", context)
