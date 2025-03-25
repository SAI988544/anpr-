import os
import uuid
import logging
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ✅ Logging Configuration
logging.basicConfig(level=logging.INFO)

# ✅ YOLO Model Path
YOLO_MODEL_PATH = os.path.join("static", "models", "licence_plate.pt")
model = YOLO(YOLO_MODEL_PATH)

# ✅ PaddleOCR Initialization
paddle_ocr = PaddleOCR(use_angle_cls=True,
                       use_space_char=True,
                       use_language_model=False,
                       lang='en',
                       show_log=False)

# ✅ Check if file is allowed
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# ✅ Save uploaded file
def save_uploaded_file(file, upload_folder):
    try:
        if file and file.filename:
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            file_path = os.path.join(upload_folder, filename)
            os.makedirs(upload_folder, exist_ok=True)
            file.save(file_path)
            logging.info(f"File saved to: {file_path}")
            return file_path
        return None
    except Exception as e:
        logging.error(f"Error saving file: {str(e)}")
        return None

# ✅ Preprocess Image for OCR
def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

# ✅ PaddleOCR Function
def run_paddle_ocr(image_np):
    try:
        result = paddle_ocr.ocr(image_np, cls=True)
        plate_text = ""
        for line in result:
            for word in line:
                plate_text += word[1][0]
        plate_text = plate_text.replace(" ", "").strip().upper()
        logging.info(f"[PaddleOCR] Detected: {plate_text}")
        return plate_text if plate_text else None
    except Exception as e:
        logging.error(f"[PaddleOCR Error] {e}")
        return None

# ✅ License Plate Detection Logic
def detect_license_plate(image_path):
    try:
        if not os.path.exists(image_path):
            logging.error(f"Image not found: {image_path}")
            return None, 0.0, None

        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image: {image_path}")
            return None, 0.0, None

        output_image = image.copy()
        results = model(image)
        detected_plate = None
        detected_confidence = 0.0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                logging.info(f"YOLO detection confidence: {conf}")

                if conf < 0.5:
                    continue

                # Draw bounding box
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Extract plate region and run OCR
                plate_roi = image[y1:y2, x1:x2]
                preprocessed_roi = preprocess_for_ocr(plate_roi)
                text_detected = run_paddle_ocr(preprocessed_roi)

                # Show OCR result on image
                if text_detected:
                    detected_plate = text_detected
                    detected_confidence = conf
                    cv2.putText(output_image, f"Plate: {detected_plate}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2)

        # Save processed image
        processed_dir = os.path.join("static", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        processed_image_path = os.path.join(processed_dir, f"processed_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(processed_image_path, output_image)
        logging.info(f"Processed image saved: {processed_image_path}")

        return detected_plate, detected_confidence, processed_image_path

    except Exception as e:
        logging.error(f"Detection failed: {str(e)}", exc_info=True)
        return None, 0.0, None
