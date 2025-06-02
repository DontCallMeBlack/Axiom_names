import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import re
from functools import lru_cache
import cv2
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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
        with open('names.csv', 'r') as f:
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
        # Convert to PIL Image if it's not already
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
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
        stride = 90  # Changed stride to 90 pixels
        x_offset = 150  # Shift right by 150 pixels
        
        # Calculate smaller window dimensions
        small_window_width = window_width // 2
        small_window_height = window_height // 2
        
        all_matches = set()
        
        # Convert image to numpy array once
        image_array = np.array(image)
        
        # Pre-compile regex patterns
        patterns = [
            re.compile(r'^(.+?)[,，]\s*(\d+)$'),
            re.compile(r'^(.+?)\s+(\d+)$'),
            re.compile(r'^(.+?)[.]\s*(\d+)$'),
            re.compile(r'^(.+?)[:：]\s*(\d+)$'),
            re.compile(r'^(.+?)[-]\s*(\d+)$'),
            re.compile(r'^(.+?)(\d+)$')
        ]
        
        # Scan vertically with stride
        for y in range(0, height - small_window_height + 1, stride):
            # Calculate window position
            start_x = x_offset
            start_y = y
            
            # Extract window directly from numpy array
            window = image_array[start_y:start_y + small_window_height, start_x:start_x + small_window_width]
            
            # Convert window to PIL Image for OCR
            window_pil = Image.fromarray(window)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(window_pil, config='--psm 6')
            text = text.strip()
            
            if text:
                # Clean up common OCR mistakes
                text = text.replace('1', 'l').replace('0', 'o')
                text = re.sub(r'\s+', ' ', text)
                
                # Try different patterns for name-level matching
                found_pattern = False
                for pattern in patterns:
                    match = pattern.match(text)
                    if match:
                        potential_name = match.group(1).strip()
                        level = match.group(2).strip()
                        
                        # Additional cleaning for the name
                        potential_name = re.sub(r'[^a-zA-Z0-9]', '', potential_name)
                        
                        # Try to match the name
                        matches = find_matching_names(potential_name)
                        if matches:
                            all_matches.update(matches)
                        
                        found_pattern = True
                        break
                
                if not found_pattern:
                    # If no pattern matched, try matching the whole text
                    matches = find_matching_names(text)
                    if matches:
                        all_matches.update(matches)
        
        return ' '.join(all_matches)
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        return ''

@lru_cache(maxsize=1000)
def find_matching_names(text, threshold=0.65):
    if not text:
        return []
    
    # Preprocess the extracted text
    text = text.lower()
    
    # Clean the text
    text = re.sub(r'^(area|search|maa|mage|druid|warrior|lirsreach|lir\'sreach|reach|druid-|mage-|warrior-)\s*', '', text, flags=re.IGNORECASE)
    text = text.strip()
    
    # Handle special cases like "OP Tank" -> "OPTank"
    if 'op' in text and 'tank' in text:
        text = 'OPTank'
    
    text = preprocess_name(text)
    
    # Skip if text is too short
    if len(text) < 3:
        return []
    
    # Try exact match first
    if text in NAMES_DICT:
        return [NAMES_DICT[text]]
    
    # Use cosine similarity for fuzzy matching
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    try:
        # Create vectors for the text and all possible matches
        all_texts = [text] + list(NAMES_DICT.keys())
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Get best match above threshold
        best_match = None
        best_similarity = 0
        
        for idx, similarity in enumerate(similarities[0]):
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = list(NAMES_DICT.values())[idx]
        
        if best_similarity >= threshold:
            return [best_match]
            
    except Exception as e:
        print(f"Error in cosine similarity: {str(e)}")
        # Fallback to partial matching with stricter requirements
        for processed_name, original_name in NAMES_DICT.items():
            if len(processed_name) > 3 and (
                processed_name in text or 
                text in processed_name or
                # Handle special cases
                (processed_name == 'azazelbreath' and 'azaze' in text) or
                (processed_name == 'escilator' and 'esci' in text)
            ):
                # Require 70% length match for partial matches
                if len(processed_name) >= len(text) * 0.7:
                    return [original_name]
    
    return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
    if '.' not in file.filename or \
       file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, MP4, AVI, MOV'}), 400
    
    if file:
        try:
            # Read the file
            file_bytes = file.read()
            
            # Check if it's an image or video
            if file.content_type.startswith('image/'):
                # Process image
                image = Image.open(io.BytesIO(file_bytes))
                matches = extract_text_from_image(image)
                
                return jsonify({
                    'matches': matches,
                    'type': 'image'
                })
                
            elif file.content_type.startswith('video/'):
                # Process video using temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = os.path.join(temp_dir, secure_filename(file.filename))
                    with open(temp_path, 'wb') as f:
                        f.write(file_bytes)
                    
                    try:
                        # Use OpenCV for video processing
                        cap = cv2.VideoCapture(temp_path)
                        all_matches = set()
                        
                        # Process every 20th frame to avoid processing too many frames
                        frame_count = 0
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            if frame_count % 20 == 0:
                                # Convert frame to PIL Image
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_pil = Image.fromarray(frame_rgb)
                                matches = extract_text_from_image(frame_pil)
                                if matches:
                                    all_matches.update(matches.split())
                            
                            frame_count += 1
                        
                        cap.release()
                    finally:
                        # Clean up is handled automatically by the context manager
                        pass
                
                return jsonify({
                    'matches': ' '.join(all_matches),
                    'type': 'video'
                })
                
        except Exception as e:
            print(f"\n=== Error ===")
            print(str(e))
            print("=============\n")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 