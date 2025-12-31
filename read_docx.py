import zipfile
import xml.etree.ElementTree as ET
import re

docx_path = r'1 Basics of python\Basics of python.docx'

# Extract text from docx
with zipfile.ZipFile(docx_path, 'r') as zip_ref:
    xml_content = zip_ref.read('word/document.xml')

# Parse XML and extract text
root = ET.fromstring(xml_content)
namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
paragraphs = root.findall('.//w:t', namespace)

text_content = []
for para in paragraphs:
    if para.text:
        text_content.append(para.text)

print('\n'.join(text_content))
