function uploadCSV() {
    const fileInput = document.getElementById("csvFile");
    const resultBox = document.getElementById("result");
    const graphsBox = document.getElementById("graphs");
    const dynamicBox = document.getElementById("dynamicGraph");
    const confidenceImg = document.getElementById("confidenceImg");

    // Reset outputs
    resultBox.innerText = "";
    graphsBox.style.display = "none";
    dynamicBox.style.display = "none";

    if (!fileInput.files || fileInput.files.length === 0) {
        resultBox.innerText = "Please select a CSV file first.";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        // Show prediction
        resultBox.innerText = "Recommended Crop: " + data.crop;

        // ✅ Show static graphs
        graphsBox.style.display = "block";

        // ✅ Show dynamic confidence graph
        if (data.confidence_graph) {
            confidenceImg.src =
                "data:image/png;base64," + data.confidence_graph;
            dynamicBox.style.display = "block";
        }
    })
    .catch(error => {
        resultBox.innerText = "Error occurred while predicting.";
    });
}

/* Dark mode toggle */
function toggleTheme() {
    document.body.classList.toggle("dark");

    const btn = document.getElementById("themeToggle");
    if (document.body.classList.contains("dark")) {
        btn.innerText = "☀️ Light Mode";
    } else {
        btn.innerText = "🌙 Dark Mode";
    }
}
