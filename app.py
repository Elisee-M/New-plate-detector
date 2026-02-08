from io import BytesIO
from base64 import b64encode
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
import easyocr

app = Flask(__name__, static_folder=None)
reader = easyocr.Reader(['en'], gpu=False)

def read_image(file_storage):
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def preprocess_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def find_plate_rect(img):
    edges = preprocess_edges(img)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    best = None
    best_area = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        ar = cw / float(ch) if ch else 0
        if area > (h * w * 0.01) and 2 <= ar <= 6:
            if area > best_area:
                best_area = area
                best = (x, y, cw, ch)
    return best

def crop_from_bbox(img, bbox):
    xs = [int(pt[0]) for pt in bbox]
    ys = [int(pt[1]) for pt in bbox]
    x0, y0, x1, y1 = max(min(xs), 0), max(min(ys), 0), max(xs), max(ys)
    h, w = img.shape[:2]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    roi = img[y0:y1, x0:x1]
    return roi if roi.size else img

def crop_plate(img):
    results = reader.readtext(img, detail=1)
    candidates = []
    for bbox, text, conf in results:
        cleaned = ''.join(ch for ch in text if ch.isalnum()).upper()
        if len(cleaned) >= 5 and conf >= 0.3:
            candidates.append((conf, cleaned, bbox))
    if candidates:
        candidates.sort(reverse=True)
        _, _, bbox = candidates[0]
        roi = crop_from_bbox(img, bbox)
    else:
        rect = find_plate_rect(img)
        if rect is None:
            roi = img
        else:
            x, y, cw, ch = rect
            roi = img[y:y+ch, x:x+cw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def ocr_text(img):
    result = reader.readtext(img, detail=0, paragraph=True)
    text = ''.join(result)
    text = ''.join(ch for ch in text if ch.isalnum()).upper()
    return text

def to_data_url(img):
    success, buf = cv2.imencode('.png', img)
    if not success:
        return None
    b64 = b64encode(buf.tobytes()).decode('ascii')
    return f'data:image/png;base64,{b64}'

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/detect', methods=['POST'])
def detect():
    f = request.files.get('image')
    if not f:
        return jsonify({'error': 'no image'}), 400
    img = read_image(f)
    if img is None:
        return jsonify({'error': 'invalid image'}), 400
    plate = crop_plate(img)
    text = ocr_text(plate)
    data_url = to_data_url(plate)
    return jsonify({'text': text, 'plate_data_url': data_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
