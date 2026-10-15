import unittest
import pandas as pd
from ma_core import deduplicate,validate_assessment,summarize,doi_key
class CoreTests(unittest.TestCase):
 def test_missing_titles_not_dropped(self):
  df=pd.DataFrame([{'title':'','doi':''},{'title':'','doi':''},{'title':'Same Title Example','doi':'10.1000/abc'},{'title':'Other','doi':'https://doi.org/10.1000/abc'}])
  a,b=deduplicate(df); self.assertEqual(len(a),3);self.assertEqual(len(b),1)
 def test_exclusion_without_quote_uncertain(self):
  x=validate_assessment({'eligibility':'exclude','evidence':'made up','outcomes':[]},['TG'],'real article',True)
  self.assertEqual(x['eligibility'],'uncertain')
 def test_unverified_numeric_not_confirmed(self):
  x=validate_assessment({'eligibility':'include','outcomes':[{'name':'TG','status':'confirmed','evidence':'invented 9.9'}]},['TG'],'Results TG 2.3',True)
  self.assertEqual(x['outcomes'][0]['status'],'uncertain')
 def test_no_abstract_confirm(self):
  x=validate_assessment({'eligibility':'include','outcomes':[{'name':'TG','status':'confirmed','evidence':'TG was 12.3'}]},['TG'],'TG was 12.3',False)
  self.assertEqual(x['outcomes'][0]['status'],'uncertain')
 def test_counts_are_not_prediction_interval(self):
  df=pd.DataFrame([{'outcome':'TG','paper_id':'A','status':'confirmed','reviewed':False},{'outcome':'TG','paper_id':'B','status':'uncertain','reviewed':False}]);r=summarize(df,10,2)
  self.assertEqual(int(r.iloc[0]['AI_confirmed_candidates']),1);self.assertEqual(int(r.iloc[0]['conservative_possible_max']),10)
 def test_doi(self):self.assertEqual(doi_key('https://doi.org/10.1234/ABC'),'10.1234/abc')
if __name__=='__main__': unittest.main()
