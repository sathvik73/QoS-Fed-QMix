import sys

try:
    import fitz
    doc = fitz.open(sys.argv[1])
    text = ""
    for page in doc:
        text += page.get_text()
    
    with open('pdf_content.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Successfully read with PyMuPDF")
except ImportError:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(sys.argv[1])
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        with open('pdf_content.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Successfully read with PyPDF2")
    except ImportError:
        print("Please install PyMuPDF (fitz) or PyPDF2")
except Exception as e:
    import traceback
    traceback.print_exc()
