"""
End-to-End Urdu OCR: YOLO text detection + UTRNet recognition.
Uses YOLOv8 (finetuned on UrduDoc) for text line detection and
UTRNet-Large for Urdu text recognition.

Requires: pip install ultralytics
Download YOLO model: https://huggingface.co/spaces/abdur75648/UrduOCR-UTRNet/resolve/main/yolov8m_UrduDoc.pt
"""
import os
import math
import argparse
import torch
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

from model import Model
from utils import CTCLabelConverter
from dataset import NormalizePAD

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def text_recognizer(img_cropped, model, converter, device):
    """Recognize Urdu text in a cropped line image using UTRNet."""
    img = img_cropped.convert('L')
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(32 * ratio) > 400:
        resized_w = 400
    else:
        resized_w = math.ceil(32 * ratio)
    img = img.resize((resized_w, 32), Image.Resampling.BICUBIC)
    transform = NormalizePAD((1, 32, 400))
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)])
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    return preds_str


def load_recognition_model(model_path, device):
    """Load UTRNet recognition model and converter."""
    with open("UrduGlyphs.txt", "r", encoding="utf-8") as f:
        content = f.readlines()
        content = ''.join([str(elem).strip('\n') for elem in content])
    content = content + " "

    converter = CTCLabelConverter(content)

    class Opt:
        pass
    opt = Opt()
    opt.num_class = len(converter.character)
    opt.device = device
    opt.FeatureExtraction = 'HRNet'
    opt.SequenceModeling = 'DBiLSTM'
    opt.Prediction = 'CTC'
    opt.num_fiducial = 20
    opt.input_channel = 1
    opt.output_channel = 32
    opt.hidden_size = 256
    opt.imgH = 32
    opt.imgW = 400
    opt.batch_max_length = 100

    recognition_model = Model(opt)
    recognition_model = recognition_model.to(device)
    recognition_model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    recognition_model.eval()
    return recognition_model, converter


def process_document(image_path, output_file='output.txt',
                     recognition_model_path='saved_models/UTRNet-Large/best_norm_ED.pth',
                     detection_model_path='yolov8m_UrduDoc.pt',
                     conf=0.2, save_debug=True):
    """Process a document image: detect text lines and recognize Urdu text."""
    device = torch.device('cpu')
    print(f"Device: {device}")

    recognition_model, converter = load_recognition_model(
        recognition_model_path, device
    )
    print(f"Recognition model loaded: {recognition_model_path}")

    detection_model = YOLO(detection_model_path)
    print(f"Detection model loaded: {detection_model_path}")

    input_img = Image.open(image_path)
    input_img = ImageOps.exif_transpose(input_img)
    input_img = input_img.convert('RGB')
    print(f"Processing: {image_path} ({input_img.size[0]}x{input_img.size[1]})")

    detection_results = detection_model.predict(
        source=input_img, conf=conf, imgsz=1280,
        save=False, nms=True, device=device
    )
    bounding_boxes = detection_results[0].boxes.xyxy.cpu().numpy().tolist()
    bounding_boxes.sort(key=lambda x: x[1])
    print(f"Detected {len(bounding_boxes)} text lines")

    if save_debug:
        debug_img = input_img.copy()
        draw = ImageDraw.Draw(debug_img)
        import numpy as np
        for i, box in enumerate(bounding_boxes):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            draw.rectangle(box, fill=None, outline=color, width=3)
            draw.text((box[0], box[1] - 15), f"Line {i}", fill=color)
        debug_img.save("detected_lines.png")
        print("Debug image saved: detected_lines.png")

    os.makedirs("line_crops", exist_ok=True)
    for f in os.listdir("line_crops"):
        os.remove(os.path.join("line_crops", f))

    texts = []
    for i, box in enumerate(bounding_boxes):
        cropped = input_img.crop(box)
        cropped.save(f"line_crops/line_{i:03d}.png")

        with torch.no_grad():
            text = text_recognizer(cropped, recognition_model, converter, device)
        texts.append(text.strip())
        print(f"  Line {i}: {text.strip()}")

    output_path = os.path.abspath(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Source: {image_path}\n")
        f.write(f"Lines detected: {len(bounding_boxes)}\n")
        f.write(f"{'=' * 50}\n\n")
        for text in texts:
            f.write(text + '\n')

    print(f"\nOutput saved to: {output_path}")
    combined = '\n'.join(texts)
    print(f"\n{'=' * 50}")
    print("Combined recognized text:")
    print(f"{'=' * 50}")
    print(combined)
    return texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-End Urdu OCR")
    parser.add_argument('--image_path', required=True, help="Path to document image")
    parser.add_argument('--output_file', default='output.txt', help="Output text file")
    parser.add_argument('--recognition_model',
                        default='saved_models/UTRNet-Large/best_norm_ED.pth')
    parser.add_argument('--detection_model', default='yolov8m_UrduDoc.pt')
    parser.add_argument('--conf', type=float, default=0.2,
                        help="YOLO confidence threshold")
    parser.add_argument('--save_debug', action='store_true', default=True,
                        help="Save detected_lines.png debug image")
    args = parser.parse_args()

    process_document(
        image_path=args.image_path,
        output_file=args.output_file,
        recognition_model_path=args.recognition_model,
        detection_model_path=args.detection_model,
        conf=args.conf,
        save_debug=args.save_debug,
    )
