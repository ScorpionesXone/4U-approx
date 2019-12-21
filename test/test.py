from numpy.testing import assert_allclose, assert_array_equal

import matplotlib.pyplot as plt
import numpy as np
import math as m

import sys
sys.path.append('..')
from freddi import Freddi, State, EvolutionResults

from numpy import exp

from lmfit import minimize, Parameters
from scipy import interpolate

import doctest
import unittest

from rytp import residual
from rytp import run

DAY = 86400
_Xi_max = 40
_T_iC = 1e8
Mx = 2e33*9.4
GM = 6.673e-8 * Mx
Mdotout = 0
Cirr = 2.9e-4
kerr =  0.4
Mopt = 2.5e33
period = 1.116*DAY
q = Mx/Mopt

routI = 2.25e11
MdotinI = 1e19
alphaI = 0.75

default_kwargs = dict(wind=b'no', Mdotout=Mdotout, Mx=Mx, kerr = kerr,
            initialcond=b'quasistat', powerorder=1, opacity=b'OPAL',  boundcond = b'Tirr', Thot = 1e4, 
            Cirr = Cirr, time=35*DAY, tau=0.35*DAY, Nx=10000, gridscale=b'linear')
    
life = run(wind=b'__Woods_Which_Shields__', windparams=[_Xi_max, _T_iC], F0=MdotinI*np.sqrt(GM*routI), alpha = alphaI, rout = routI)
Sin = life.evolve()


random.seed(888)

x1 = Sin.t/DAY + 3.5
y1 = Sin.Mdot_in
yerr1 = random.normalvariate(0, 5e17)

params = Parameters()
params.add('rout', min= 1e11, value = 1.7e11, max = 4e11)
params.add('alpha', min = 0.1, value = 0.4, max = 1.6)
params.add('Mdotin', min = 1e18, value = 1.4e19, max = 1e20)
params.add('x0', min= 1.0, value = 3.5, max = 5.0)
params['x0'].vary = False

Burst = minimize(residual, params, args=(x1*DAY, y1, yerr1 ,'wind no tidal'))

Dalpha = Burst.params['alpha'].value
DMdotin = Burst.params['Mdotin'].value
Drout = Burst.params['rout'].value

print(Dalpha, DMdotin, Drout)


class MyTestCase(unittest.TestCase):
    def test_alpha(self):
        assert_allclose(Dalpha, alphaI, rtol=1e-1)

    def test_rout(self):
        assert_allclose(Drout, routI, rtol=1e-1)
    
    def test_Mdout(self):
        assert_allclose(DMdotin, MdotinI, rtol=1e-1)

if __name__ == '__main__':
    unittest.main()
