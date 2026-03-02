async function uploadPDF() {
    const fileInput = document.getElementById("pdfFile");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select a PDF file");
        return;
    }

    const loading = document.getElementById("loading");
    const results = document.getElementById("results");

    const formData = new FormData();
    formData.append("file", file);

    loading.classList.remove("hidden");
    results.classList.add("hidden");

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Server error");
        }

        const data = await response.json();

        // ===== Fill Results =====

        // Summary
        const formattedSummary = data.summary
            .split(/\n+/)
            .map(p => `<p>${p.trim()}</p>`)
            .join("");
        document.getElementById("summary").innerHTML = formattedSummary;

        // Classification
        const outcomeText = document.getElementById("outcome");
        outcomeText.innerText = data.classification.prediction;
        outcomeText.style.color =
            data.classification.prediction === "ALLOWED"
                ? "green"
                : "red";

        document.getElementById("confidence").innerText =
            (data.classification.confidence * 100).toFixed(2) + "%";

        // Risk
        document.getElementById("risk").innerText =
            data.regression.risk_score.toFixed(4);

        // Reason
        document.getElementById("reason").innerText = data.reason;

        // Stats
        document.getElementById("words").innerText = data.text_stats.words;
        document.getElementById("sentences").innerText = data.text_stats.sentences;

        results.classList.remove("hidden");

    } catch (error) {
        alert("Something went wrong while analyzing the document.");
        console.error(error);
    } finally {
        // 🔥 THIS GUARANTEES LOADER ALWAYS HIDES
        loading.classList.add("hidden");
    }
}