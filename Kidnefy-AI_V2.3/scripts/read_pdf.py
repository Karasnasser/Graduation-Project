import PyPDF2
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

reader = PyPDF2.PdfReader(r'data/raw/GP_DN_Ch1-4 (LV).pdf')
print(f'Total pages: {len(reader.pages)}')

with open('pdf_content.txt', 'w', encoding='utf-8') as f:
    for i in range(len(reader.pages)):
        text = reader.pages[i].extract_text()
        f.write(f'--- PAGE {i+1} ---\n')
        f.write(text)
        f.write('\n\n')

print('Done! Content saved to pdf_content.txt')
