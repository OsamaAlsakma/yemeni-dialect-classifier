/* ============================================================
   classifier.js — Отправка аудио на API и отображение результатов
   ============================================================ */

let selectedFile = null;

document.addEventListener("DOMContentLoaded", () => {
    initFileUpload();
    initClassifyButton();
});

/**
 * Инициализация загрузки файла через drag-and-drop и клик
 */
function initFileUpload() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");

    // Клик по зоне открывает диалог выбора файла
    dropZone.addEventListener("click", () => fileInput.click());

    // Изменение через диалог
    fileInput.addEventListener("change", e => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag-and-drop
    dropZone.addEventListener("dragover", e => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", e => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
}

/**
 * Обработка выбора файла
 */
function handleFileSelect(file) {
    // Проверка типа
    if (!file.type.startsWith("audio/")) {
        showError("Выберите аудиофайл (WAV, MP3, FLAC, OGG, M4A)");
        return;
    }

    selectedFile = file;

    // Отображение информации о файле
    const fileInfo = document.getElementById("fileInfo");
    fileInfo.querySelector(".file-name").textContent = file.name;
    fileInfo.querySelector(".file-size").textContent = formatFileSize(file.size);
    fileInfo.style.display = "flex";

    // Активация кнопки
    document.getElementById("classifyBtn").disabled = false;
    hideError();
}

/**
 * Инициализация кнопки классификации
 */
function initClassifyButton() {
    const classifyBtn = document.getElementById("classifyBtn");
    classifyBtn.addEventListener("click", performClassification);
}

/**
 * Выполняет классификацию: отправляет файл на API и отображает результат
 */
async function performClassification() {
    // Определяем источник аудио: загруженный файл или записанный с микрофона
    let audioData;
    let fileName;

    const recordedBlob = (typeof getRecordedBlob === "function")
        ? getRecordedBlob() : null;

    if (selectedFile) {
        audioData = selectedFile;
        fileName  = selectedFile.name;
    } else if (recordedBlob) {
        audioData = recordedBlob;
        fileName  = "recorded.webm";
    } else {
        showError("Сначала загрузите файл или запишите голос");
        return;
    }

    // Формируем multipart-форму
    const formData = new FormData();
    formData.append("audio", audioData, fileName);
    formData.append("top_n", "3");

    // Показываем индикатор загрузки
    hideError();
    document.getElementById("loading").style.display = "block";
    document.getElementById("results").style.display = "none";
    document.getElementById("classifyBtn").disabled = true;

    try {
        // Отправляем POST-запрос на API
        const response = await fetch("/api/v1/classify/", {
            method: "POST",
            body: formData,
            headers: {
                "X-CSRFToken": getCookie("csrftoken") || "",
            },
        });

        const data = await response.json();

        // Скрываем индикатор
        document.getElementById("loading").style.display = "none";
        document.getElementById("classifyBtn").disabled = false;

        if (!response.ok || !data.success) {
            const errMsg = data.error || "Ошибка обработки запроса";
            const details = data.details ? `\n${data.details}` : "";
            showError(errMsg + details);
            return;
        }

        // Отображаем результаты
        displayResults(data.result);

    } catch (err) {
        document.getElementById("loading").style.display = "none";
        document.getElementById("classifyBtn").disabled = false;
        showError("Сетевая ошибка: " + err.message);
    }
}

/**
 * Отображает результаты классификации
 */
function displayResults(result) {
    // Главный результат
    document.getElementById("topDialect").textContent = result.top_dialect_ru;
    document.getElementById("topRegion").textContent  = result.top_region_ru;
    document.getElementById("topConfidence").textContent =
        result.top_confidence_pct.toFixed(1) + " %";

    // Top-N список
    const topNList = document.getElementById("topNList");
    topNList.innerHTML = "";

    result.top_n.forEach((entry, i) => {
        const item = document.createElement("div");
        item.className = "top-n-item";
        item.innerHTML = `
            <div class="top-n-rank">${i + 1}.</div>
            <div class="top-n-info">
                <div class="dialect-ru">${entry.dialect_ru}</div>
                <div class="region">${entry.region_ru}</div>
                <div class="top-n-progress">
                    <div class="top-n-bar" style="width: ${entry.confidence_pct}%;"></div>
                </div>
            </div>
            <div class="top-n-pct">${entry.confidence_pct.toFixed(1)} %</div>
        `;
        topNList.appendChild(item);
    });

    // Информация о модели
    document.getElementById("modelUsed").textContent = result.model_used.toUpperCase();

    // Показываем секцию результатов
    document.getElementById("results").style.display = "block";

    // Плавный скролл к результатам
    document.getElementById("results").scrollIntoView({ behavior: "smooth" });
}

/**
 * Показывает сообщение об ошибке
 */
function showError(message) {
    const panel = document.getElementById("errorPanel");
    document.getElementById("errorMessage").textContent = message;
    panel.style.display = "block";
}

function hideError() {
    document.getElementById("errorPanel").style.display = "none";
}
