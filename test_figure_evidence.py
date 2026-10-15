import unittest
from figure_evidence import interpret_error, classify_plot, parse_digitized_csv, validate_two_group
class FigureEvidenceTests(unittest.TestCase):
    def test_sd(self):
        a=interpret_error(10,10.2,9.8,error_type='SD',n=8)
        self.assertEqual(a['status'],'candidate'); self.assertAlmostEqual(a['sd'],.2)
    def test_sem(self):
        self.assertAlmostEqual(interpret_error(10,10.1,9.9,error_type='SEM',n=9)['sd'],.3)
    def test_unknown(self): self.assertIsNone(interpret_error(10,11,error_type='UNKNOWN')['sd'])
    def test_asymmetric(self): self.assertIsNone(interpret_error(10,11,9.5,error_type='SD')['sd'])
    def test_ci(self): self.assertGreater(interpret_error(10,11,9,error_type='CI95',n=12)['sd'],0)
    def test_box(self): self.assertIn('assumptions',classify_plot('box-and-whisker'))
    def test_invalid_n(self): self.assertIsNone(interpret_error(10,11,error_type='SEM',n=1)['sd'])
    def test_sufficiency(self): self.assertTrue(validate_two_group(10,2,8,9,1,8)[0])
    def test_csv(self): self.assertEqual(len(parse_digitized_csv(b'X,Y\n1,2\n3,4\n')),2)
if __name__=='__main__':unittest.main()
