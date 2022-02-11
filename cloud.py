import numpy as np
import matplotlib.pyplot as plt

import astropy.units as au
import astropy.constants as ac

class Cloud(object):
    """
    Simple class for spherical clouds

    Initialize by giving two of M, R, or Sigma (in units of M_sun, pc)
    """
    
    mH = ac.u*1.008          # Mass of a hydrogen atom
    # Particle mass per hydrogen atom (in fully atomic gas)
    muH = 1.4*mH.to(au.gram)

    def __init__(self, M=None, R=None, Sigma=None, alpha_vir=2.0):
        if M is not None and Sigma is not None and R is not None:
            raise ValueError('Exactly two of M, R, Sigma must be defined')
        # Check if input is dimensionless
        if M is not None and not isinstance(M, au.quantity.Quantity):
            M = (M*ac.M_sun).to(au.M_sun)
        if R is not None and not isinstance(R, au.quantity.Quantity):
            R = (R*ac.pc).to(au.pc)
        if Sigma is not None and not isinstance(Sigma, au.quantity.Quantity):
            Sigma = (Sigma*ac.M_sun/ac.pc**2).to(ac.M_sun/ac.pc**2)
        if M is not None and Sigma is not None:  # M and Sigma are given
            self.M = M
            self.Sigma = Sigma
            self.R = np.sqrt(M/(np.pi*Sigma))
        elif M is not None and R is not None:  # M and R are given
            self.M = M
            self.R = R
            self.Sigma = M/(np.pi*R**2)

        self.M = self.M.to(au.Msun)
        self.R = self.R.to(au.pc)
        self.Sigma = self.Sigma.to(au.Msun/au.pc**2)

        self.rho = Cloud.calc_rho(self.M, self.R).to(au.g/au.cm**3)
        self.nH = (self.rho/Cloud.muH).to(au.cm**(-3))
        self.tff = Cloud.calc_tff(self.rho)

        self.alpha_vir = alpha_vir
        self.vesc = (np.sqrt((2.0*ac.G*self.M)/self.R)).to(au.km/au.s)
        self.sigma1d = np.sqrt(self.alpha_vir/10.0)*self.vesc

    def __str__(self):
        if not self.M.shape:
            return 'Cloud object: M={0:<5g}, R={1:<5g},'\
                   'Sigma={2:<5g}, nH={3:<5g}, tff={4:<5g}'.format(
                       self.M, self.R, self.Sigma, self.nH, self.tff)
        else:
            return 'Cloud objects with shape {0:s}'.format(self.M.shape)

    @staticmethod
    def calc_rho(M, R):
        return M/(4.0*np.pi/3.0*R**3)

    @staticmethod
    def calc_tff(rho):
        return np.sqrt((3.0*np.pi)/(32.0*ac.G*rho)).to(au.Myr)

    @staticmethod
    def calc_Sigma_from_M_vesc(M, vesc):
        return ((vesc**2/(2.0*ac.G))**2/(np.pi*M)).to(au.M_sun/au.pc**2)

    @staticmethod
    def calc_Sigma_from_M_tff0(M, tff0):
        return ((np.pi*M/64.0)**(1.0/3.0)/(tff0**2*ac.G)**(2.0/3.0)).to(au.M_sun/au.pc**2)

