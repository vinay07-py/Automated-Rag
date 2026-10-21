// frontend/static/js/neo_ds_ui.js

// Function handling the secure chunked multi-gigabyte upload pipeline
async function uploadLargeDataset() {
    const fileInput = document.getElementById('file-upload-input'); // matches your HTML ID
    if (!fileInput.files.length) return alert("Please select a valid dataset target matrix.");

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    updateTerminalUI("[SYSTEM] Initializing memory-safe data transmission array...");

    // Send data straight to our FastAPI async handler node
    const response = await fetch("/api/upload", {
        method: "POST",
        body: formData
    });

    const result = await response.json();
    console.log(result.message);

    // Start interval loop checking to update active agent progress states on the web page screen
    setInterval(checkPipelineStatus, 1000);
}

// Polls the server status every second to get logs and updates without freezing the browser window
async function checkPipelineStatus() {
    const response = await fetch("/api/status");
    const data = await response.json();

    // Appends running console logs straight into your cyberpunk console box interface components
    const consoleBox = document.getElementById('terminal-console-output');
    if (consoleBox) {
        consoleBox.innerHTML = data.logs.join("\n");
        consoleBox.scrollTop = consoleBox.scrollHeight; // Auto scroll down
    }

    // Renders final generated analysis markdown into the visualization section element frame container
    if (data.status === "Complete" && data.report) {
        document.getElementById('report-display-pane').innerHTML = data.report;
    }
}

// Integrated Local RAG chat interface execution loop
async function sendChatQuery() {
    const chatInput = document.getElementById('rag-user-input');
    const queryText = chatInput.value;
    if (!queryText) return;

    appendChatLog("👤 User", queryText);
    chatInput.value = "";

    const response = await fetch("/api/rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText })
    });

    const data = await response.json();
    appendChatLog("🤖 Cognitive Agent", data.response);
}

function updateTerminalUI(text) {
    const consoleBox = document.getElementById('terminal-console-output');
    if (consoleBox) consoleBox.innerHTML += `\n${text}`;
}

function appendChatLog(sender, text) {
    const chatContainer = document.getElementById('chat-history-container');
    if (chatContainer) {
        chatContainer.innerHTML += `<p><b>${sender}:</b> ${text}</p>`;
    }
}