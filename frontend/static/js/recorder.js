/* ============================================================
   recorder.js — Запись аудио с микрофона через MediaRecorder API
   ============================================================ */

// Состояние записи
let mediaRecorder = null;
let audioChunks   = [];
let isRecording   = false;
let recordedBlob  = null;

document.addEventListener("DOMContentLoaded", () => {
    const recordBtn = document.getElementById("recordBtn");
    if (recordBtn) {
        recordBtn.addEventListener("click", toggleRecording);
    }
});

/**
 * Переключение записи: старт или остановка
 */
async function toggleRecording() {
    if (!isRecording) {
        await startRecording();
    } else {
        stopRecording();
    }
}

/**
 * Начинает запись с микрофона пользователя
 */
async function startRecording() {
    try {
        // Запрос разрешения на доступ к микрофону
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Создаём MediaRecorder для захвата аудио
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        // Обработчики событий
        mediaRecorder.addEventListener("dataavailable", e => {
            if (e.data.size > 0) {
                audioChunks.push(e.data);
            }
        });

        mediaRecorder.addEventListener("stop", () => {
            // Сборка финального blob после остановки
            recordedBlob = new Blob(audioChunks, { type: "audio/webm" });

            // Отображение в плеере
            const audioElem = document.getElementById("recordedAudio");
            audioElem.src = URL.createObjectURL(recordedBlob);
            audioElem.style.display = "block";

            // Освобождение микрофона
            stream.getTracks().forEach(t => t.stop());

            // Активация кнопки классификации
            document.getElementById("classifyBtn").disabled = false;
        });

        mediaRecorder.start();
        isRecording = true;
        updateRecordButton();
    } catch (err) {
        alert("Не удалось получить доступ к микрофону: " + err.message);
    }
}

/**
 * Останавливает запись
 */
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        updateRecordButton();
    }
}

/**
 * Обновляет внешний вид кнопки записи
 */
function updateRecordButton() {
    const btn = document.getElementById("recordBtn");
    const label = btn.querySelector(".record-label");
    const status = document.getElementById("recordStatus");

    if (isRecording) {
        btn.classList.add("recording");
        label.textContent = "■ Остановить запись";
        status.textContent = "🔴 Идёт запись...";
    } else {
        btn.classList.remove("recording");
        label.textContent = "Начать запись";
        status.textContent = "✓ Запись готова. Нажмите Классифицировать.";
    }
}

/**
 * Возвращает записанный аудио-blob или null
 */
function getRecordedBlob() {
    return recordedBlob;
}
