# AI PDF Filler

AI PDF Filler is a Python application that automatically fills out PDF forms using AI-powered text generation. It detects empty fields in PDF documents and fills them with contextually appropriate responses based on provided information.

## Features

- GUI interface for easy file selection and processing
- Automated detection of empty form fields
- AI-powered text generation for filling out forms
- Support for handling multiple-cell inputs
- Question and Answer section processing
- PDF to image conversion for processing
- Converts processed images back to PDF format

## Prerequisites

Before running the application, ensure you have:

1. Python 3.7 or higher installed
2. Tesseract OCR installed on your system

### Installing Tesseract OCR

#### Windows
1. Download the installer from the [UB-Mannheim Tesseract page](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer
3. Add the Tesseract installation directory to your system PATH
4. Update the Tesseract path in `testingcode.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### macOS
```bash
brew install tesseract
```

#### Linux
```bash
sudo apt-get install tesseract-ocr
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-pdf-filler.git
cd ai-pdf-filler
```

2. Run the installation script to install required Python packages:
```bash
python install.py
```

This will install the following dependencies:
- opencv-python
- numpy
- PyMuPDF
- pytesseract
- Pillow
- transformers
- torch
- setuptools

## Usage

1. Run the GUI application:
```bash
python gui.py
```

2. Using the GUI:
   - Select your input PDF file
   - Choose the context text file containing relevant information
   - Select an output directory for the processed files
   - Click "Start Processing" to begin

## Project Structure

- `gui.py` - Main GUI application
- `testingcode.py` - Core processing logic
- `install.py` - Package installation script

## How It Works

1. The application converts PDF pages to high-resolution images
2. It detects empty form fields using image processing techniques
3. For each empty field:
   - Identifies the field name
   - Generates appropriate text using AI
   - Fills the field with the generated text
4. Processes Q&A sections similarly
5. Converts the processed images back to PDF format

## Requirements

See `install.py` for a complete list of Python package requirements.

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Acknowledgments

- Tesseract OCR for text recognition
- OpenCV for image processing
- PyMuPDF for PDF handling
