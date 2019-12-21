import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy import integrate

import sys
sys.path.append('..')
from freddi import Freddi, State, EvolutionResults

DAY = 86400


from numpy import exp, sin

from lmfit import minimize, Parameters
from scipy import interpolate


_Xi_max = 40
_T_iC = 1e8
Mx = 2e33*9.4
GM = 6.673e-8 * Mx
Mdotout = 0
Cirr = 2.9e-4
kerr =  0.4
Mopt = 2.5e33
period = 1.116*DAY
#rout = 1.7e11
q = Mx/Mopt

a = np.genfromtxt('asu.tsv', names = True)
x1 = (a['tend']/2 + a['tbegin']/2) - (2452443.31221716/2 + 2452443.32503197/2)
y1 = a['dotM']*1e18
yerr1 = [a['dotM']-a['b_dotM'],a['B_dotM']- a['dotM']]


def residual(params, x, data, eps_data, windrad):
    """Function to be minimized

    Keyword arguments:
    params -- Parameters involved in th fitting
    x -- The set of values of the moments of observations
    data -- Source luminosity at these moments
    eps_data -- Observation errors
    windrad -- windradarameter for determining the model used
    """
    rout = params['rout']
    alpha = params['alpha']
    Mdotin = params['Mdotin']
    x0 = params['x0']

    
    default_kwargs = dict(wind=b'no', Mdotout=Mdotout, Mx=Mx, kerr = kerr, alpha = alpha,
            initialcond=b'quasistat', powerorder=1, opacity=b'OPAL', boundcond = b'Tirr', Thot = 1e4, 
            Cirr = Cirr, time=35*DAY, tau=0.3*DAY, Nx=10000, gridscale=b'linear')
    
    def run(**input_kwargs):
        kwargs = default_kwargs.copy()
        kwargs.update(input_kwargs)
        fr = Freddi(**kwargs)
        return fr
    
    
    if windrad == 'wind and tidal':
        frwT = run(wind=b'__Woods_Which_Shields__' , windparams=[_Xi_max, _T_iC], Mdotin = Mdotin, rout = None, Mopt = Mopt, period = period)
        resultT  = frwT.evolve()

        track = interpolate.splrep(resultT.t + x0*DAY, resultT.Mdot_in)
        model = interpolate.splev(x1*DAY, track)
         
    elif windrad == 'wind no tidal':
        frw = run(rout = rout, F0=Mdotin*np.sqrt(GM*rout), wind=b'__Woods_Which_Shields__' , windparams=[_Xi_max, _T_iC])
        result  = frw.evolve()

        track = interpolate.splrep(result.t + x0*DAY, result.Mdot_in)
        model = interpolate.splev(x1*DAY, track)
                   
    elif windrad == 'no wind and tidal':
        fr0T = run(Mdotin = Mdotin, rout = None, Mopt = Mopt, period = period)
        r0T  = fr0T.evolve()

        track = interpolate.splrep(r0T.t + x0*DAY, r0T.Mdot_in)
        model = interpolate.splev(x1*DAY, track)
    elif windrad == 'no wind no tidal':
        fr0 = run(rout = rout, F0=Mdotin*np.sqrt(GM*rout))
        r0  = fr0.evolve()
        
        track = interpolate.splrep(r0.t + x0*DAY, r0.Mdot_in)
        model = interpolate.splev(x1*DAY, track)
    else:
        print('Something is wrong')
    
    return (data-model) / eps_data

params = Parameters()
params.add('rout', min= 1e11, value = 1.7e11, max = 4e11)
params.add('alpha', min = 0.1, value = 0.4, max = 1.6)
params.add('Mdotin', min = 1e18, value = 1.4e19, max = 1e20)
params.add('x0', min= 1.0, value = 3.5, max = 5.0)
params['x0'].vary = False

out = minimize(residual, params, args=(x1*DAY, y1, yerr1 ,'wind no tidal'))
out.params

params['rout'].vary = False

ouT = minimize(residual, params, args=(x1*DAY, y1, yerr1, 'wind and tidal'))
ouT.params

ouT0 = minimize(residual, params, args=(x1*DAY, y1, yerr1 ,'no wind and tidal'))
ouT0.params


#params['alpha'].min =  1.2
#params['alpha'].value =  1.3
#params['Mdotin'].min =  1e19
#params['alpha'].value =  1e19
#out0 = minimize(residual, params, args=(x1*DAY, y1, yerr1 ,'no wind no tidal'))
#out0.params


def Cov(Soul):
    """Determination of the covariance matrix

    Keyword arguments:
    Soul --The result of minimization
    """
    S = Soul.covar.copy()
    i, j = np.indices(S.shape)
    Mind = S / np.sqrt(S[i, i] * S[j, j])
    return Mind
    
print(Cov(out)) 


alpha = out.params['alpha'].value
Mdotin = out.params['Mdotin'].value
rout = out.params['rout'].value

alphaT = ouT.params['alpha'].value
MdotinT = ouT.params['Mdotin'].value


alpha0T = ouT0.params['alpha'].value
Mdotin0T = ouT0.params['Mdotin'].value

#alpha0 = out0.params['alpha'].value
#Mdotin0 = out0.params['Mdotin'].value
#x00= out0.params['x0'].value

default_kwargs = dict(wind=b'no', Mdotout=Mdotout, Mx=Mx, kerr = kerr,
            initialcond=b'quasistat', powerorder=1, opacity=b'OPAL',  boundcond = b'Tirr', Thot = 1e4, 
            Cirr = Cirr, time=35*DAY, tau=0.35*DAY, Nx=10000, gridscale=b'linear')
    
def run(**input_kwargs):
    """Element filler

    Keyword arguments:
    **input_kwargs --Input parameter list
    """
    kwargs = default_kwargs.copy()
    kwargs.update(input_kwargs)
    fr = Freddi(**kwargs)
    return fr
    
#fr0 = run(F0=Mdotin0*np.sqrt(GM*rout), alpha = alpha0, rout = rout)
#r0  = fr0.evolve()

frT0 = run(alpha = alpha0T, Mdotin = Mdotin0T, rout = None, Mopt = Mopt, period = period)
rT0  = frT0.evolve()

frw = run(wind=b'__Woods_Which_Shields__', windparams=[_Xi_max, _T_iC], F0=Mdotin*np.sqrt(GM*rout), alpha = alpha, rout = rout)
result = frw.evolve()

frwT = run(wind=b'__Woods_Which_Shields__', windparams=[_Xi_max, _T_iC], Mdotin = MdotinT, alpha = alphaT, rout = None, Mopt = Mopt, period = period)
resultT = frwT.evolve()


alphaT1 = round(alphaT, 3)
alpha0T1 = round(alpha0T, 3)

plt.figure(figsize = (10,6))
plt.title(r'Wind off: $\alpha = $' + str(alphaT1)+r', ' + r'Wind on: $\alpha = $' + str(alpha0T1) + r'; ' + r'Woods Approx Case', fontsize=16)
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r'$t$, days after peak', fontsize=33)
plt.ylabel(r'$\dotM$, g/s', fontsize=33)
plt.errorbar(x1, y1, yerr1, fmt='x', color = 'k', label='Observe')
plt.plot(rT0.t/DAY + 3.5, rT0.Mdot_in, label='Wind is off')
plt.plot(resultT.t / DAY + 3.5, resultT.Mdot_in, label='Wind is on')  
plt.axhline(np.exp(-1)*resultT.Mdot_in[0], ls='-.', color='k', lw=0.5, label='$\mathrm{e}^{-1}$')
plt.legend()
plt.grid()
#plt.savefig('MdotvsTime.pdf', bbox_inches = 'tight')


plt.figure(figsize = (10,6))
plt.title(r'Wind off: $\alpha = $' + str(alphaT1)+r', ' + r'Wind on: $\alpha = $' + str(alpha0T1) + r'; ' + r'Woods Approx Case', fontsize=16)
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r'$t$, days after peak', fontsize=33)
plt.ylabel(r'$R_{\rm hot}$, cm', fontsize=33) 
plt.plot(rT0.t[1:]/DAY + 3.5, rT0.last_R[1:], label='Wind is off', color = 'r') 
plt.errorbar(resultT.t[1:] / DAY + 3.5, resultT.last_R[1:], fmt='*', color = 'k', label='Wind is on') 
plt.legend()
plt.grid()
#plt.savefig('RoutvsTime.pdf', bbox_inches = 'tight')


plt.figure(figsize = (10,6))
plt.xlabel(r'$t$, days after peak', fontsize=33)
plt.ylabel(r'$\dot{M}_{\rm wind}/\dot{M}_{\rm acc}$', fontsize=33)
plt.plot(resultT.t[1:] / DAY+ 3.5, resultT.Mdot_wind[1:]/resultT.Mdot_in[1:], color = 'g')
plt.grid()
#plt.savefig('RelatvsTime.pdf', bbox_inches = 'tight')


m_P = 1.6726e-24
k_B = 1.3807e-16
mu = 0.61
Ric = ((GM*mu*m_P)/(k_B*_T_iC))
SMTH = -resultT.windC[1,:]*GM*GM/(4*m.pi*(resultT.h[1,:])*(resultT.h[1,:])*(resultT.h[1,:]))

plt.figure(figsize = (10,6))
plt.xlim(0, resultT.last_R[2]/Ric)
plt.xlabel(r'$R/R_{\rm IC}$', fontsize=33)
plt.ylabel(r'$W$, g/(s*cm$^{2}$)', fontsize=33)
plt.plot(resultT.R[1,:]/Ric, SMTH)
plt.grid()
#plt.savefig('WindvsRad.pdf', bbox_inches = 'tight')

