/**
 * PDF-to-LaTeX Web UI — Main Application
 */

const API_BASE = window.location.origin;

// ---- State ----
const state = {
    taskId: null,
    ws: null,
    file: null,
    polling: null,
};

// ---- DOM Elements ----
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ---- Initialization ----
document.addEventListener("DOMContentLoaded", () => {
    initUploadZone();
    checkModelsStatus();
    setInterval(checkModelsStatus, 30000);
});

// ---- Upload ----
function initUploadZone() {
    const zone = $("#upload-zone");
    const input = $("#file-input");

    zone.addEventListener("dragover", (e) => {
        e.preventDefault();
        zone.classList.add("drag-over");
    });

    zone.addEventListener("dragleave", () => {
        zone.classList.remove("drag-over");
    });

    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelected(e.dataTransfer.files[0]);
        }
    });

    input.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelected(e.target.files[0]);
        }
    });
}

function handleFileSelected(file) {
    const supported = [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"];
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!supported.includes(ext)) {
        showToast(`Unsupported file type: ${ext}`, "error");
        return;
    }

    state.file = file;
    const info = $("#file-info");
    info.classList.add("visible");
    info.querySelector(".filename").textContent = file.name;
    info.querySelector(".filesize").textContent = formatSize(file.size);

    $("#btn-convert").disabled = false;
    addLog(`File selected: ${file.name} (${formatSize(file.size)})`, "info");
}

async function startConversion() {
    if (!state.file) return;

    const btn = $("#btn-convert");
    btn.disabled = true;
    btn.textContent = "Uploading...";

    try {
        const formData = new FormData();
        formData.append("file", state.file);

        const resp = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || "Upload failed");
        }

        const data = await resp.json();
        state.taskId = data.task_id;

        addLog(`Task created: ${data.task_id}`, "success");
        showProgressSection(true);
        connectWebSocket(data.task_id);
        startPolling(data.task_id);

        btn.textContent = "Processing...";
    } catch (err) {
        showToast(`Upload failed: ${err.message}`, "error");
        addLog(`Error: ${err.message}`, "error");
        btn.disabled = false;
        btn.textContent = "Convert to LaTeX";
    }
}

// ---- WebSocket ----
function connectWebSocket(taskId) {
    if (state.ws) {
        state.ws.close();
    }

    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${wsProto}//${window.location.host}/ws/task/${taskId}`;

    try {
        state.ws = new WebSocket(wsUrl);

        state.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateProgress(data);
        };

        state.ws.onerror = () => {
            addLog("WebSocket error, falling back to polling", "warn");
        };

        state.ws.onclose = () => {
            state.ws = null;
        };
    } catch {
        addLog("WebSocket not available, using polling", "warn");
    }
}

// ---- Polling Fallback ----
function startPolling(taskId) {
    if (state.polling) clearInterval(state.polling);

    state.polling = setInterval(async () => {
        try {
            const resp = await fetch(`${API_BASE}/api/task/${taskId}/status`);
            if (resp.ok) {
                const data = await resp.json();
                updateProgress(data);

                if (data.stage === "complete" || data.stage === "failed") {
                    clearInterval(state.polling);
                    state.polling = null;
                }
            }
        } catch {
            // Silently retry
        }
    }, 2000);
}

// ---- Progress Updates ----
function updateProgress(data) {
    const pct = Math.round((data.progress || 0) * 100);
    $("#progress-bar").style.width = `${pct}%`;
    $("#progress-pct").textContent = `${pct}%`;
    $("#progress-msg").textContent = data.message || "";

    // Stage badge
    const badge = $("#stage-badge");
    badge.textContent = formatStage(data.stage);

    if (data.stage === "complete") {
        badge.className = "stage-badge complete";
        onConversionComplete(data);
    } else if (data.stage === "failed") {
        badge.className = "stage-badge failed";
        onConversionFailed(data);
    } else {
        badge.className = "stage-badge processing";
    }
}

function formatStage(stage) {
    const map = {
        init: "Initializing",
        preprocessing: "Preprocessing",
        layout_analysis: "Layout Analysis",
        recognition: "Recognition",
        assembly: "Assembly",
        validation: "Validation",
        complete: "Complete",
        failed: "Failed",
    };
    return map[stage] || stage;
}

async function onConversionComplete(data) {
    addLog("Conversion complete!", "success");
    showToast("Conversion complete!", "success");

    $("#btn-convert").textContent = "Convert to LaTeX";
    $("#btn-convert").disabled = false;

    // Fetch LaTeX result
    try {
        const resp = await fetch(`${API_BASE}/api/task/${state.taskId}/result`);
        if (resp.ok) {
            const result = await resp.json();
            const editor = $("#latex-editor");
            editor.value = result.latex;
            addLog(`LaTeX code loaded (${result.latex.length} chars)`, "info");
        }
    } catch (err) {
        addLog(`Failed to fetch result: ${err.message}`, "error");
    }

    // Show action buttons
    showActionButtons(true);
}

function onConversionFailed(data) {
    addLog(`Conversion failed: ${data.error || "Unknown error"}`, "error");
    showToast("Conversion failed", "error");

    $("#btn-convert").textContent = "Retry";
    $("#btn-convert").disabled = false;
}

// ---- Action Buttons ----
function showActionButtons(show) {
    const btns = $("#action-buttons");
    btns.style.display = show ? "flex" : "none";
}

function showProgressSection(show) {
    const section = $("#progress-section");
    section.classList.toggle("visible", show);
    $("#log-panel").classList.toggle("visible", show);
}

async function downloadTex() {
    if (!state.taskId) return;

    try {
        const resp = await fetch(`${API_BASE}/api/task/${state.taskId}/result`);
        if (resp.ok) {
            const data = await resp.json();
            downloadText(data.latex, `${state.taskId}_output.tex`, "application/x-tex");
            showToast("LaTeX file downloaded", "success");
        }
    } catch (err) {
        showToast(`Download failed: ${err.message}`, "error");
    }
}

async function downloadPdf() {
    if (!state.taskId) return;

    try {
        const resp = await fetch(`${API_BASE}/api/task/${state.taskId}/pdf`);
        if (resp.ok) {
            const blob = await resp.blob();
            downloadBlob(blob, `${state.taskId}_output.pdf`);
            showToast("PDF downloaded", "success");
        } else {
            showToast("PDF not available (compiler may not be installed)", "info");
        }
    } catch (err) {
        showToast(`Download failed: ${err.message}`, "error");
    }
}

async function downloadZip() {
    if (!state.taskId) return;

    try {
        const resp = await fetch(`${API_BASE}/api/task/${state.taskId}/download`);
        if (resp.ok) {
            const blob = await resp.blob();
            downloadBlob(blob, `${state.taskId}_output.zip`);
            showToast("ZIP archive downloaded", "success");
        }
    } catch (err) {
        showToast(`Download failed: ${err.message}`, "error");
    }
}

async function recompile() {
    if (!state.taskId) return;

    const code = $("#latex-editor").value;
    if (!code.trim()) {
        showToast("No LaTeX code to compile", "error");
        return;
    }

    addLog("Recompiling LaTeX...", "info");

    try {
        const resp = await fetch(`${API_BASE}/api/task/${state.taskId}/recompile`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code }),
        });

        const data = await resp.json();
        if (data.success) {
            addLog("Recompilation successful!", "success");
            showToast("Compiled successfully", "success");
        } else {
            addLog(`Compilation error: ${data.log.substring(0, 300)}`, "error");
            showToast("Compilation failed — see log", "error");
        }
    } catch (err) {
        showToast(`Recompile failed: ${err.message}`, "error");
    }
}

function copyLatex() {
    const editor = $("#latex-editor");
    navigator.clipboard.writeText(editor.value).then(() => {
        showToast("LaTeX copied to clipboard", "success");
    });
}

// ---- Model Status ----
async function checkModelsStatus() {
    try {
        const resp = await fetch(`${API_BASE}/api/models/status`);
        if (!resp.ok) {
            setStatusDot("offline");
            return;
        }
        const data = await resp.json();

        setStatusDot(data.ollama_available ? "online" : "offline");
        $("#status-text").textContent = data.ollama_available ? "API Online" : "API Offline";

        // Update model grid
        updateModelItem("ollama-status", data.ollama_available ? "Online" : "Offline");
        updateModelItem("compiler-status", data.latex_compiler_available ? "Available" : "Not Found");
        updateModelItem("detikzify-status", data.detikzify_loaded ? "Loaded" : "Ready");
        updateModelItem("models-count", data.ollama_models ? data.ollama_models.length.toString() : "0");
    } catch {
        setStatusDot("offline");
        $("#status-text").textContent = "API Offline";
    }
}

function setStatusDot(status) {
    const dot = $("#status-dot");
    dot.className = `status-dot ${status}`;
}

function updateModelItem(id, value) {
    const el = $(`#${id}`);
    if (el) el.textContent = value;
}

// ---- Log ----
function addLog(message, level = "info") {
    const panel = $("#log-panel");
    panel.classList.add("visible");

    const time = new Date().toLocaleTimeString();
    const entry = document.createElement("div");
    entry.className = `log-entry ${level}`;
    entry.textContent = `[${time}] ${message}`;
    panel.appendChild(entry);
    panel.scrollTop = panel.scrollHeight;
}

// ---- Toast ----
function showToast(message, type = "info") {
    const container = $("#toast-container");
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ---- Utilities ----
function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
}

function downloadText(text, filename, mime) {
    const blob = new Blob([text], { type: mime });
    downloadBlob(blob, filename);
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
