const runButton = document.getElementById("runButton");
const labelInput = document.getElementById("labelInput");
const stepsInput = document.getElementById("stepsInput");
const strategyInput = document.getElementById("strategyInput");
const leftConnector = document.getElementById("lineUsEu");
const rightConnector = document.getElementById("lineEuAsia");
const expertLeft = document.querySelector(".tpu-pin");
const expertRight = document.querySelector(".gpu-pin");
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

function restartPulse(element) {
  console.log("restartPulse", element?.id || element?.className || element);
  element.classList.remove("active");
  void element.offsetWidth;
  element.classList.add("active");
  console.log("restartPulse classes", element?.className || element);
}

function flashExpert(expertId) {
  console.log("flashExpert", expertId, {
    leftConnector,
    rightConnector,
    expertLeft,
    expertRight,
  });
  clearGlow();
  if (expertId === 0) {
    restartPulse(leftConnector);
    restartPulse(expertLeft);
  } else if (expertId === 1) {
    restartPulse(rightConnector);
    restartPulse(expertRight);
  }

  window.setTimeout(clearGlow, 180);
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
    console.log("demo event payload", payload);
    drawFrame(payload.frame);

    if (payload.type === "start") {
      clearGlow();
      return;
    }

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
