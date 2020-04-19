import numpy as np
from scipy.stats import norm
from numpy.testing import assert_allclose
from ..src.bandwidth import bw_scott, norm_pdf


data = np.linspace(-5, 5, 10000)

def test_bw_scott():
    assert bw_scott(data) > 0

def test_norm_pdf():
    assert_allclose(norm_pdf(data), norm.pdf(data))
