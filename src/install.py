import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    'opencv-python',
    'numpy',
    'PyMuPDF',
    'pytesseract',
    'Pillow',
    'transformers',
    'torch',
    'setuptools'  # Adding setuptools which includes pkg_resources
]

print("Installing required packages...")

for package in packages:
    print(f"Installing {package}...")
    install(package)

print("\nAll Python packages have been installed.")

print("\nIMPORTANT: You also need to install Tesseract OCR on your system.")
print("Installation instructions:")
print("- Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki")
print("- macOS: Use Homebrew: brew install tesseract")
print("- Linux: Use your package manager, e.g., sudo apt-get install tesseract-ocr")

print("\nAfter installing Tesseract OCR, make sure to update the path in your code:")
print("pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Adjust this path as needed")

print("\nInstallation process complete.")