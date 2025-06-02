# OCR Text Recognition App

This Flask application performs OCR (Optical Character Recognition) on images and videos to extract and match names.

## Deployment to Vercel

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy the project:
```bash
vercel
```

4. For production deployment:
```bash
vercel --prod
```

## Project Structure
- `app.py`: Main Flask application
- `requirements.txt`: Python dependencies
- `vercel.json`: Vercel configuration
- `names.csv`: Database of names for matching
- `templates/`: HTML templates

## Environment Variables
No environment variables are required for basic functionality.

## Dependencies
All dependencies are listed in `requirements.txt` and will be automatically installed during deployment.

## Notes
- The application uses EasyOCR for text recognition
- Processing time may be slow on first request due to model loading
- Maximum file size is 16MB
- Supported file types: PNG, JPG, JPEG, MP4, AVI, MOV

## Features

- Image and video processing
- OCR text extraction using EasyOCR
- Name matching with fuzzy search
- Support for multiple file formats (PNG, JPG, JPEG, MP4, AVI, MOV)
- Web interface for easy file upload

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/axiom-names.git
cd axiom-names
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `names.csv` file with the list of names to match against (comma-separated).

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

1. Open your web browser and navigate to `http://localhost:5000`
2. Upload an image or video file
3. The application will process the file and return any matching names

## Requirements

- Python 3.8+
- Flask
- EasyOCR
- OpenCV
- scikit-learn
- Other dependencies listed in requirements.txt

## License

MIT License 