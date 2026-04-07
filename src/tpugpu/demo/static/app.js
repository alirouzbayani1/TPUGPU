const runButton = document.getElementById("runButton");
const labelInput = document.getElementById("labelInput");
const stepsInput = document.getElementById("stepsInput");
const leftConnector = document.getElementById("lineUsEu");
const rightConnector = document.getElementById("lineEuAsia");
const expertLeft = document.querySelector(".tpu-pin");
const expertRight = document.querySelector(".gpu-pin");
const routerPin = document.querySelector(".router-pin");
const mapWrap = document.querySelector(".map-wrap");
const canvas = document.getElementById("stateCanvas");
const ctx = canvas.getContext("2d");

let currentSource = null;

function pinCenter(pinElement) {
  const dot = pinElement.querySelector(".pin-dot");
  const mapRect = mapWrap.getBoundingClientRect();
  const dotRect = dot.getBoundingClientRect();
  const x = dotRect.left - mapRect.left + dotRect.width / 2;
  const y = dotRect.top - mapRect.top + dotRect.height / 2;
  return { x, y };
}

function updateConnectionLines() {
  const left = pinCenter(expertLeft);
  const router = pinCenter(routerPin);
  const right = pinCenter(expertRight);

  leftConnector.setAttribute("x1", `${left.x}`);
  leftConnector.setAttribute("y1", `${left.y}`);
  leftConnector.setAttribute("x2", `${router.x}`);
  leftConnector.setAttribute("y2", `${router.y}`);

  rightConnector.setAttribute("x1", `${router.x}`);
  rightConnector.setAttribute("y1", `${router.y}`);
  rightConnector.setAttribute("x2", `${right.x}`);
  rightConnector.setAttribute("y2", `${right.y}`);

  console.log("connection-debug", {
    pinCenters: {
      tpu: left,
      router,
      gpu: right,
    },
    lineUsEu: {
      x1: leftConnector.getAttribute("x1"),
      y1: leftConnector.getAttribute("y1"),
      x2: leftConnector.getAttribute("x2"),
      y2: leftConnector.getAttribute("y2"),
    },
    lineEuAsia: {
      x1: rightConnector.getAttribute("x1"),
      y1: rightConnector.getAttribute("y1"),
      x2: rightConnector.getAttribute("x2"),
      y2: rightConnector.getAttribute("y2"),
    },
  });
}

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
    strategy: "alternating",
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
window.addEventListener("resize", updateConnectionLines);
window.addEventListener("load", updateConnectionLines);
updateConnectionLines();
drawFrame(new Array(32 * 32).fill(0));
