/* ============================================================
   main.js — Общая логика интерфейса
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
});

/**
 * Инициализация переключения вкладок (Загрузка / Запись)
 */
function initTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            // Снимаем активность со всех
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            // Активируем выбранную
            btn.classList.add("active");
            document.getElementById(`tab-${target}`).classList.add("active");
        });
    });
}

/**
 * Получает CSRF-токен из cookies Django
 */
function getCookie(name) {
    let value = null;
    if (document.cookie && document.cookie !== "") {
        const cookies = document.cookie.split(";");
        for (const c of cookies) {
            const cookie = c.trim();
            if (cookie.substring(0, name.length + 1) === (name + "=")) {
                value = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return value;
}

/**
 * Форматирует размер файла в КБ или МБ
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " КБ";
    return (bytes / (1024 * 1024)).toFixed(2) + " МБ";
}
