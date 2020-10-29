"""
These are different functions describing how labour is affected by increased WBGT.
I recommend reading doi.org/10.2760/07911
"""
import numpy as np
from scipy.special import erfc

SQRT2 = np.sqrt(2)

def labour_li(WBGT):
    """
        Labour loss function from Li et al doi.org/10.1016/j.buildenv.2015.09.005
        It quite optimistic, 100% loss occurs at a very high WBGT.
    """
    return np.clip(100 - ((-0.57 * WBGT) + 106.16), 0, 100)


def labour_dunne(WBGT):
    """
        Labour loss function from Dunne et al doi.org/10.1038/nclimate1827
        Based on safe working standards, so somewhat pessimistic.
    """
    return np.clip(100 - (100-(25*(np.maximum(0, WBGT-25))**(2/3))), 0, 100)


kfl_300w_loc, kfl_300w_scale = 33.49, 3.94
kfl_200w_loc, kfl_200w_scale = 35.53, 3.94
kfl_400w_loc, kfl_400w_scale = 32.47, 4.16

def labour_kfl_200w(WBGT):
    """Apply a labour loss model to an iris cube of WBGT values
    Currently specified to match the synthetic model used in the Lancet Countdown 2019
    Input:  T1, WBGT values
    Return:  A number between 0 and 1,
    specifying the efficiency of physical labour at that temperature"""
    x = np.divide(
        np.subtract(WBGT, kfl_200w_loc),
        kfl_200w_scale * -SQRT2)
    return np.clip(erfc(x)/2*100, 0, 100)

def labour_kfl_300w(WBGT):
    """Apply a labour loss model to an iris cube of WBGT values
    Currently specified to match the synthetic model used in the Lancet Countdown 2019
    Input:  T1, WBGT values
    Return:  A number between 0 and 1,
    specifying the efficiency of physical labour at that temperature"""
    x = np.divide(
        np.subtract(WBGT, kfl_300w_loc),
        kfl_300w_scale * -SQRT2)
    return np.clip(erfc(x)/2*100, 0, 100)

def labour_kfl_400w(WBGT):
    """Apply a labour loss model to an iris cube of WBGT values
    Currently specified to match the synthetic model used in the Lancet Countdown 2019
    Input:  T1, WBGT values
    Return:  A number between 0 and 1,
    specifying the efficiency of physical labour at that temperature"""
    x = np.divide(
        np.subtract(WBGT, kfl_400w_loc),
        kfl_400w_scale * -SQRT2)
    return np.clip(erfc(x)/2*100, 0, 100)


def labour_sahu(WBGT):
    """Labour loss according to Sahu et al doi.org/10.2486/indhealth.2013-0006
    This is based on observations of rice harvesting.
    Is quite optimistic, 100% loss occurs at a very high WBGT.
    """
    return np.clip(100-((-5.14*WBGT) + 218), 0, 100)

