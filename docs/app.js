// 그리기 캔버스, 버튼, 결과 표시를 담당하는 메인 스크립트

import { preprocessCanvas } from "./preprocess.js";
import { loadModel, predict } from "./model.js";

// 펜 두께 (데스크톱 버전과 동일하게 280px 캔버스 기준 18px)
const PEN_WIDTH = 18;

const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button");
const predictionValue = document.getElementById("prediction-value");
const statusText = document.getElementById("status-text");
const probList = document.getElementById("prob-list");

let isDrawing = false;
let lastPoint = null;
let hasDrawing = false;

// 0~9 확률 막대 항목을 미리 만들어 둠
const probItems = [];
for (let digit = 0; digit < 10; digit++) {
  const item = document.createElement("li");
  item.className = "prob-item";
  item.innerHTML = `
    <span class="prob-digit">${digit}</span>
    <div class="prob-bar"><div class="prob-fill"></div></div>
    <span class="prob-value">0.0%</span>`;
  probList.appendChild(item);
  probItems.push({
    item,
    fill: item.querySelector(".prob-fill"),
    value: item.querySelector(".prob-value"),
  });
}

// 캔버스를 검은 배경으로 초기화
function resetCanvas() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  hasDrawing = false;
}

// 화면 좌표를 캔버스 내부 좌표(280x280)로 변환 (캔버스가 CSS로 축소돼도 정확하게)
function getCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

// 두 점 사이에 둥근 흰색 선 그리기
function drawLine(from, to) {
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = PEN_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  hasDrawing = true;
}

// Pointer Events: 마우스, 터치, 펜을 하나의 코드로 처리
canvas.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  // 캔버스 밖으로 나가도 획을 계속 추적 (일부 환경에서 실패해도 그리기는 계속되도록 예외 처리)
  try {
    canvas.setPointerCapture(event.pointerId);
  } catch (error) {
    console.warn("Pointer capture failed:", error);
  }
  isDrawing = true;
  lastPoint = getCanvasPoint(event);
  drawLine(lastPoint, lastPoint); // 클릭만 해도 점이 찍히도록
});

canvas.addEventListener("pointermove", (event) => {
  if (!isDrawing) return;
  const point = getCanvasPoint(event);
  drawLine(lastPoint, point);
  lastPoint = point;
});

function endStroke() {
  if (!isDrawing) return;
  isDrawing = false;
  lastPoint = null;
  // 한 획이 끝날 때마다 자동 인식
  recognize();
}

canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);

clearButton.addEventListener("click", () => {
  resetCanvas();
  showResult(null);
});

// 예측 결과를 화면에 표시 (probs가 null이면 초기 상태로)
function showResult(probs) {
  if (!probs) {
    predictionValue.textContent = "?";
    probItems.forEach(({ item, fill, value }) => {
      item.classList.remove("best");
      fill.style.width = "0%";
      value.textContent = "0.0%";
    });
    return;
  }
  const best = probs.indexOf(Math.max(...probs));
  predictionValue.textContent = String(best);
  probItems.forEach(({ item, fill, value }, digit) => {
    const percent = probs[digit] * 100;
    item.classList.toggle("best", digit === best);
    fill.style.width = `${percent}%`;
    value.textContent = `${percent.toFixed(1)}%`;
  });
}

// 불러온 모델 (로딩이 끝나기 전에는 null)
let model = null;

// 인식 함수: 캔버스 → 전처리 → 모델 계산 → 결과 표시
function recognize() {
  if (!hasDrawing) return;
  if (!model) {
    statusText.textContent = "Model is still loading...";
    return;
  }
  const input = preprocessCanvas(canvas);
  if (!input) return;
  showResult(predict(model, input));
}

// 초기화: 캔버스를 비우고 모델을 비동기로 불러옴
resetCanvas();
showResult(null);
statusText.textContent = "Loading model...";
loadModel()
  .then((loaded) => {
    model = loaded;
    const accuracy = (loaded.testAccuracy * 100).toFixed(2);
    statusText.textContent = `Model ready (test accuracy ${accuracy}%). Draw a digit.`;
    recognize(); // 로딩 중에 이미 그린 그림이 있으면 바로 인식
  })
  .catch((error) => {
    console.error(error);
    statusText.textContent = `Failed to load model: ${error.message}`;
  });
