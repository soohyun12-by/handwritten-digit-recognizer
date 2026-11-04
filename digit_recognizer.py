"""Draw a digit with the mouse and let the trained model recognize it."""

import os
import tkinter as tk

import joblib
import numpy as np
from PIL import Image, ImageDraw

from train_model import MODEL_PATH, train_and_save

# 캔버스 크기와 펜 두께 설정
CANVAS_SIZE = 280
PEN_WIDTH = 18


def preprocess(image):
    """Convert a drawn PIL image into a 784-length MNIST-style vector."""
    # 그레이스케일 numpy 배열로 변환
    pixels = np.array(image, dtype=np.float32)

    # 아무것도 그리지 않았으면 None 반환
    rows = np.any(pixels > 0, axis=1)
    cols = np.any(pixels > 0, axis=0)
    if not rows.any():
        return None

    # 1) 숫자가 그려진 영역(bounding box)만 잘라내기
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    cropped = image.crop((left, top, right + 1, bottom + 1))

    # 2) 비율을 유지하면서 긴 변이 20픽셀이 되도록 축소 (MNIST와 동일한 방식)
    width, height = cropped.size
    scale = 20.0 / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cropped.resize(new_size, Image.LANCZOS)

    # 3) 28x28 검은 배경의 중앙에 붙이기
    canvas = Image.new("L", (28, 28), 0)
    offset = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    canvas.paste(resized, offset)
    digit = np.array(canvas, dtype=np.float32)

    # 4) 무게중심이 이미지 중앙(14, 14)에 오도록 이동
    total = digit.sum()
    ys, xs = np.indices(digit.shape)
    center_y = (ys * digit).sum() / total
    center_x = (xs * digit).sum() / total
    shift_y = int(round(14 - center_y))
    shift_x = int(round(14 - center_x))
    digit = np.roll(digit, shift_y, axis=0)
    digit = np.roll(digit, shift_x, axis=1)

    # 0~1 범위로 정규화하고 1차원 벡터로 펼치기
    return (digit / 255.0).reshape(1, -1)


class DigitApp:
    """Tkinter application with a drawing canvas and prediction display."""

    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.last_point = None
        root.title("Handwritten Digit Recognizer")
        root.resizable(False, False)

        # 화면에 보이는 그리기 캔버스 (검은 배경)
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # 모델 입력용으로 동시에 그려지는 PIL 이미지 (화면 캔버스와 동일한 내용)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        # 마우스 이벤트 연결: 드래그로 그리기, 버튼을 떼면 자동 인식
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # 버튼 영역
        tk.Button(root, text="Predict", width=12, command=self.predict).grid(row=1, column=0, pady=5)
        tk.Button(root, text="Clear", width=12, command=self.clear).grid(row=1, column=1, pady=5)

        # 결과 표시 라벨
        self.result_label = tk.Label(root, text="Draw a digit (0-9)", font=("Arial", 24, "bold"))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=5)
        self.detail_label = tk.Label(root, text="", font=("Consolas", 11), justify="left")
        self.detail_label.grid(row=3, column=0, columnspan=2, pady=(0, 10))

    def on_press(self, event):
        # 선의 시작점 기록 및 점 하나 찍기
        self.last_point = (event.x, event.y)
        self.draw_line(event.x, event.y, event.x, event.y)

    def on_drag(self, event):
        # 이전 점과 현재 점을 이어서 선 그리기
        if self.last_point:
            x0, y0 = self.last_point
            self.draw_line(x0, y0, event.x, event.y)
        self.last_point = (event.x, event.y)

    def on_release(self, _event):
        # 한 획이 끝날 때마다 실시간으로 인식
        self.last_point = None
        self.predict()

    def draw_line(self, x0, y0, x1, y1):
        """Draw the same stroke on both the screen canvas and the PIL image."""
        r = PEN_WIDTH // 2
        self.canvas.create_line(x0, y0, x1, y1, fill="white", width=PEN_WIDTH, capstyle=tk.ROUND, smooth=True)
        self.draw.line((x0, y0, x1, y1), fill=255, width=PEN_WIDTH)
        # 선의 끝을 둥글게 만들기 위해 원도 함께 그림
        self.draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=255)

    def clear(self):
        # 캔버스와 이미지를 모두 초기화
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, CANVAS_SIZE, CANVAS_SIZE), fill=0)
        self.result_label.config(text="Draw a digit (0-9)")
        self.detail_label.config(text="")

    def predict(self):
        # 전처리 후 모델로 예측
        vector = preprocess(self.image)
        if vector is None:
            self.result_label.config(text="Canvas is empty")
            return
        probs = self.model.predict_proba(vector)[0]
        best = int(np.argmax(probs))
        self.result_label.config(text=f"Prediction: {best}")

        # 확률이 높은 상위 3개 후보 표시
        top3 = np.argsort(probs)[::-1][:3]
        lines = [f"{d}: {probs[d] * 100:5.1f}%" for d in top3]
        self.detail_label.config(text="\n".join(lines))


def load_model():
    """Load the saved model, training a new one if it does not exist."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # 저장된 모델이 없으면 새로 학습
    print("No saved model found. Training a new one...")
    return train_and_save()


if __name__ == "__main__":
    app_model = load_model()
    window = tk.Tk()
    DigitApp(window, app_model)
    window.mainloop()
