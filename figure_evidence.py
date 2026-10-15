"""Human-verifiable chart extraction and effect-size feasibility helpers.
Independent implementation; no WebPlotDigitizer code is bundled.
"""
from __future__ import annotations
import io, math, re
import pandas as pd

ERROR_TYPES = ('SD', 'SEM', 'SE', 'CI95', 'CI', 'IQR', 'RANGE', 'UNKNOWN')

def _number(value):
    try:
        if value is None or str(value).strip() == '': return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError): return None

def interpret_error(mean, upper, lower=None, *, error_type='UNKNOWN', n=None, ci_level=0.95, symmetric=False):
    """Convert calibrated bar/XY plot coordinates to SD only when justified.
    Coordinates are DATA VALUES, not pixels. For symmetric=True, upper/lower are
    positive error MAGNITUDES rather than plotted endpoints.
    """
    m, u, lo, sample = map(_number, (mean, upper, lower, n))
    typ = str(error_type).strip().upper()
    result = {'mean':m, 'upper':u, 'lower':lo, 'n':sample, 'error_type':typ,
              'sd':None, 'status':'review', 'reason':''}
    if m is None or u is None:
        result['reason']='Mean and upper error endpoint are required'; return result
    if symmetric:
        if u < 0 or (lo is not None and lo < 0):
            result['reason']='Error magnitudes must be non-negative'; return result
        plus, minus = u, (lo if lo is not None else u)
    else:
        if u < m or (lo is not None and lo > m):
            result['reason']='Error endpoints are inconsistent with the mean'; return result
        plus, minus = u-m, (m-lo if lo is not None else u-m)
    result['error_plus']=plus; result['error_minus']=minus
    if typ not in ERROR_TYPES:
        result['reason']='Unrecognized error type'; return result
    if typ == 'UNKNOWN':
        result['reason']='Caption/legend must identify SD, SEM or CI'; return result
    if typ in ('IQR','RANGE'):
        result['reason']='IQR/range cannot be equated with SD without an explicit validated conversion'; return result
    if typ in ('SD','SEM','SE','CI','CI95') and abs(plus-minus) > max(1e-9,0.05*max(plus,minus)):
        result['reason']='Asymmetric error bars: do not force a symmetric SD'; return result
    radius=(plus+minus)/2
    if typ=='SD': sd=radius
    elif typ in ('SEM','SE'):
        if sample is None or sample < 2 or sample != int(sample):
            result['reason']='SEM to SD requires an integer group n >= 2'; return result
        sd=radius*math.sqrt(sample)
    else:
        if ci_level != 0.95:
            result['reason']='Only 95% confidence intervals are supported'; return result
        if sample is None or sample < 2 or sample != int(sample):
            result['reason']='CI to SD requires an integer group n >= 2'; return result
        from scipy.stats import t
        critical=t.ppf(.975,int(sample)-1)
        sd=radius*math.sqrt(sample)/critical
    result['sd']=sd; result['status']='candidate'; result['reason']='Digitized value requires verification against original figure and caption'
    return result

def classify_plot(kind):
    """Keep box-and-whisker plots separate from mean/error plots."""
    v=str(kind).casefold()
    if 'box' in v or 'quartile' in v or 'whisker' in v:
        return 'boxplot_requires_distributional_assumptions'
    if 'bar' in v or 'mean' in v or 'xy' in v or 'point' in v:
        return 'mean_error_candidate'
    return 'unknown_review'

def parse_digitized_csv(data: bytes) -> pd.DataFrame:
    """Accept WPD exported XY/bar coordinates; do not infer error type from CSV."""
    for encoding in ('utf-8-sig','cp949','latin1'):
        try:
            df=pd.read_csv(io.BytesIO(data),encoding=encoding,sep=None,engine='python')
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else: raise ValueError('CSV could not be parsed')
    if df.empty: raise ValueError('CSV contains no points')
    df.columns=[str(c).strip() for c in df.columns]
    # Preserve original headers, including WPD-specific metadata.
    return df

def validate_two_group(mean_t, sd_t, n_t, mean_c, sd_c, n_c):
    vals=[_number(v) for v in (mean_t,sd_t,n_t,mean_c,sd_c,n_c)]
    if any(v is None for v in vals): return False,'Incomplete two-arm continuous outcome data'
    mt,st,nt,mc,sc,nc=vals
    if min(st,sc)<=0: return False,'SD must be positive; zero/negative SD requires review'
    if nt<2 or nc<2 or nt!=int(nt) or nc!=int(nc):return False,'Group sample sizes must be integers >=2'
    return True,'Sufficient numerical inputs for two-group continuous-effect calculation; eligibility and independence still require review'
