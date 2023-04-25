function classifyVideo() {
    const videoFile = document.getElementById("video-file").files[0];
    if (!videoFile) {
      alert("Please select a video file.");
      return;
    }
    const formData = new FormData();
    formData.append("video-file", videoFile);
    fetch("/classify-video", {
      method: "POST",
      body: formData
    })
    .then(response => response.json())
    .then(result => {
      const resultElement = document.getElementById("result");
      resultElement.innerHTML = `
        <p>Prediction: ${result.prediction}</p>
        <p>Probability: ${result.probability}</p>
      `;
    })
    .catch(error => {
      alert(`An error occurred: ${error}`);
    });
  }
  