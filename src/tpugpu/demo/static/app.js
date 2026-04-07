const runButton = document.getElementById("runButton");
const labelInput = document.getElementById("labelInput");
const stepsInput = document.getElementById("stepsInput");
const strategyInput = document.getElementById("strategyInput");
const stepValue = document.getElementById("stepValue");
const selectedValue = document.getElementById("selectedValue");
const progressValue = document.getElementById("progressValue");
const policyValue = document.getElementById("policyValue");
const leftConnector = document.getElementById("leftConnector");
const rightConnector = document.getElementById("rightConnector");
const expertLeft = document.getElementById("expertLeft");
const expertRight = document.getElementById("expertRight");
const routerCard = document.getElementById("routerCard");
const canvas = document.getElementById("stateCanvas");
const ctx = canvas.getContext("2d");

let currentSource = null;

function drawFrame(flatPixels) {
  const imageData = ctx.createImageData(32, 32);
  for (let i = 0; i < flatPixels.length; i += 1) {
    const value = flatPixels[i];
    const offset = i * 4;
    imageData.data[offset] = value;
    imageData.data[offset + 1] = value;
    imageData.data[offset + 2] = value;
    imageData.data[offset + 3] = 255;
  }

  const offscreen = document.createElement("canvas");
  offscreen.width = 32;
  offscreen.height = 32;
  offscreen.getContext("2d").putImageData(imageData, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
}

function clearGlow() {
  leftConnector.classList.remove("active");
  rightConnector.classList.remove("active");
  expertLeft.classList.remove("active");
  expertRight.classList.remove("active");
  routerCard.classList.remove("active");
}

function flashExpert(expertId) {
  clearGlow();
  routerCard.classList.add("active");
  if (expertId === 0) {
    leftConnector.classList.add("active");
    expertLeft.classList.add("active");
  } else if (expertId === 1) {
    rightConnector.classList.add("active");
    expertRight.classList.add("active");
  }

  window.setTimeout(clearGlow, 120);
}

function startDemo() {
  if (currentSource) {
    currentSource.close();
  }

  const params = new URLSearchParams({
    label: labelInput.value,
    steps: stepsInput.value,
    strategy: strategyInput.value,
  });
  const source = new EventSource(`/api/demo/stream?${params.toString()}`);
  currentSource = source;
  runButton.disabled = true;
  runButton.textContent = "Running...";

  source.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    drawFrame(payload.frame);

    if (payload.type === "start") {
      stepValue.textContent = `0 / ${payload.steps}`;
      selectedValue.textContent = "-";
      progressValue.textContent = "0%";
      policyValue.textContent = payload.strategy;
      clearGlow();
      return;
    }

    stepValue.textContent = `${payload.step ?? payload.steps} / ${payload.steps}`;
    selectedValue.textContent = payload.selected_expert === null ? "-" : `Expert ${payload.selected_expert === 0 ? "A" : "B"}`;
    progressValue.textContent = `${Math.round(payload.progress * 100)}%`;

    if (payload.selected_expert !== null) {
      flashExpert(payload.selected_expert);
    }

    if (payload.type === "done") {
      runButton.disabled = false;
      runButton.textContent = "Run Live Demo";
      source.close();
    }
  };

  source.onerror = () => {
    runButton.disabled = false;
    runButton.textContent = "Run Live Demo";
    source.close();
  };
}

runButton.addEventListener("click", startDemo);
drawFrame(new Array(32 * 32).fill(0));
