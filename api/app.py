import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import io
import re
from functools import lru_cache
import difflib

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Set the tesseract_cmd to the absolute path of tesseract-OCR/tesseract.exe
pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tesseract-OCR', 'tesseract.exe')

# Cache for name matching
@lru_cache(maxsize=1000)
def preprocess_name(name):
    # Remove special characters and convert to lowercase
    name = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
    # Handle common OCR mistakes
    name = name.replace('1', 'l')  # Convert 1 to l
    name = name.replace('0', 'o')  # Convert 0 to o
    return name

# Cache for name dictionary
@lru_cache(maxsize=1)
def load_names():
    try:
        # Try current directory first
        path = 'names.csv'
        if not os.path.exists(path):
            # Try parent directory
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'names.csv')
        with open(path, 'r') as f:
            names = f.read().strip().split(',')
        # Create a dictionary of preprocessed names to original names
        name_dict = {preprocess_name(name): name for name in names}
        return name_dict
    except Exception as e:
        print(f"Error loading names: {str(e)}")
        # Return empty dict if file can't be read
        return {}

# Initialize names dictionary
NAMES_DICT = load_names()

def extract_text_from_image(image):
    try:
        # Ensure image is a PIL Image
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        # Convert to grayscale and resize for faster processing
        image = image.convert('L')
        width, height = image.size
        if width > 1920:  # Resize large images
            ratio = 1920 / width
            new_height = int(height * ratio)
            image = image.resize((1920, new_height), Image.Resampling.LANCZOS)
        # Apply thresholding to make only white text visible
        threshold = 200  # High threshold to only keep very bright pixels
        image = image.point(lambda x: 255 if x > threshold else 0)
        width = image.width
        height = image.height
        # Define scanning parameters
        window_width = width // 4  # First quarter of the image
        window_height = 200
        stride = 70  # Set stride to 100 pixels
        x_offset = 150  # Shift right by 150 pixels
        # Calculate smaller window dimensions
        small_window_width = window_width // 2
        small_window_height = window_height // 2
        all_matches = set()
        all_text = []
        # Pre-compile regex patterns
        patterns = [
            re.compile(r'^(.+?)[,，]\s*(\d+)$'),
            re.compile(r'^(.+?)\s+(\d+)$'),
            re.compile(r'^(.+?)[.]\s*(\d+)$'),
            re.compile(r'^(.+?)[:：]\s*(\d+)$'),
            re.compile(r'^(.+?)[-]\s*(\d+)$'),
            re.compile(r'^(.+?)(\d+)$')
        ]
        # Save the processed image for visualization
        save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scanned_image.png')
        image.save(save_path)
        print(f"Saved scanned image to {save_path}")
        # Scan vertically with stride
        for y in range(0, height - small_window_height + 1, stride):
            start_x = x_offset
            start_y = y
            # Crop window from PIL image
            window_pil = image.crop((start_x, start_y, start_x + small_window_width, start_y + small_window_height))
            # Extract text using Tesseract
            text = pytesseract.image_to_string(window_pil, config='--psm 6')
            text = text.strip()
            if text:
                text = text.replace('1', 'l').replace('0', 'o')
                text = re.sub(r'\s+', '', text)  # Remove all spaces
                print(f"Extracted text: {text}")
                all_text.append(text)
        # After collecting all text, join and match
        full_extracted = ''.join(all_text)
        matches = find_matching_names(full_extracted)
        return ' '.join(sorted(set(matches))), '\n'.join(all_text)
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        return '', ''

@lru_cache(maxsize=1000)
def find_matching_names(extracted_text, cutoff=0.8):
    if not extracted_text:
        return []
    processed_text = preprocess_name(extracted_text)
    # Sort names by length descending (longest first)
    sorted_names = sorted(NAMES_DICT.items(), key=lambda x: -len(x[0]))
    found = set()
    used_spans = []
    for key, name in sorted_names:
        # Only check if this name is not a substring of a previously found name
        if key in processed_text:
            # Check if this match overlaps with any previous match
            idx = processed_text.find(key)
            overlap = False
            for start, end in used_spans:
                if (idx >= start and idx < end) or (start >= idx and start < idx + len(key)):
                    overlap = True
                    break
            if not overlap:
                found.add(name)
                used_spans.append((idx, idx + len(key)))
    return list(found)

def combine_matching_techniques(text, threshold=100):
    """
    Combine exact and fuzzy (cosine) matching for robust name detection.
    Returns a set of matched names. No substrings allowed.
    """
    matches = set()
    processed_text = preprocess_name(text)
    # Exact match
    if processed_text in NAMES_DICT:
        matches.add(NAMES_DICT[processed_text])
    return matches

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    files = request.files.getlist('file')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No selected file'}), 400
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    all_matches = set()
    all_extracted_text = []
    for file in files:
        if '.' not in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            continue  # skip invalid files
        try:
            file_bytes = file.read()
            if file.content_type.startswith('image/'):
                image = Image.open(io.BytesIO(file_bytes))
                matches, extracted_text = extract_text_from_image(image)
                if matches:
                    all_matches.update(matches.split())
                if extracted_text:
                    all_extracted_text.append(extracted_text)
        except Exception as e:
            print(f"\n=== Error ===")
            print(str(e))
            print("=============")
            continue
    if all_matches:
        return jsonify({'matches': ' '.join(sorted(all_matches)), 'extracted_text': '\n'.join(all_extracted_text), 'type': 'image'})
    else:
        return jsonify({'error': 'No valid names found', 'extracted_text': '\n'.join(all_extracted_text)}), 400

if __name__ == '__main__':
    app.run(debug=True)