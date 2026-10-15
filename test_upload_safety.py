import io
import unittest
from zipfile import ZipFile
from upload_safety import extract_pdfs, IntakeLimits, IntakeError

class Upload:
    def __init__(self,name,data): self.name,self.data=name,data
    def getvalue(self): return self.data

def zipdata(name,content):
    b=io.BytesIO()
    with ZipFile(b,'w') as z:z.writestr(name,content)
    return b.getvalue()

class Tests(unittest.TestCase):
    def test_valid_pdf(self):
        self.assertEqual(len(extract_pdfs([Upload('a.pdf',b'%PDF-1.4\nabc')])),1)
    def test_reject_nonpdf(self):
        with self.assertRaises(IntakeError):extract_pdfs([Upload('a.pdf',b'not pdf')])
    def test_reject_traversal(self):
        with self.assertRaises(IntakeError):extract_pdfs([Upload('a.zip',zipdata('../x.pdf',b'%PDF-1.4'))])
    def test_reject_large(self):
        with self.assertRaises(IntakeError):extract_pdfs([Upload('a.pdf',b'%PDF-'+b'x'*100)],IntakeLimits(max_pdf_bytes=10))
    def test_zip(self):
        self.assertEqual(extract_pdfs([Upload('a.zip',zipdata('nested/a.pdf',b'%PDF-1.4'))])[0][0],'a.pdf')
    def test_count_limit(self):
        with self.assertRaises(IntakeError):extract_pdfs([Upload('a.pdf',b'%PDF-1.4'),Upload('b.pdf',b'%PDF-1.4')],IntakeLimits(max_pdfs=1))

if __name__=='__main__':unittest.main()
