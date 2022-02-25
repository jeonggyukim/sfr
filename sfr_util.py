import os.path as osp

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

import astropy.constants as ac
import astropy.units as au
from cloud import Cloud

# Observed SFR (Licquia & Newman 2015)
SFRtot_obs = 1.65

# Galactocentric distance of the solar circle
# Gravity Collaboration et al. (2019)
Rgal0_def = 8.178

# Mean molecular weight per H2
# Based on updated Asplund (2021) solar abundance
muH2_def = 2.809

# Mass of hydrogen atom
mH=1.008*ac.u

# Coefficient used for calculating eps_ff
b_KOF21 = 4.06*np.pi/40.0**0.5
b_P12 = 1.6*np.sqrt(3.0)*np.pi/40.0**0.5

# Unit conversion factors
avir_conv = ((1.0*au.km/au.s)**2*au.pc/(ac.G*au.M_sun)).to('').value

# Resolution of Dame et al. survey is 8.5 argmin, so rbeam = 2.47 (d/kpc) pc
rbeam_conv = (8.5*au.arcmin).cgs.value*(1.0*au.kpc).to('pc').value

# Solid angle of a single pixel in MD17
dOmega = 4.76e-6

# Core-to-star efficiency
eps_cs_def = 0.3

###########################################
# Functions calculating abundance gradient
###########################################
def get_Z_MW_MD22(Rgal, Rgal0=Rgal0_def):
    """Z as a function of Rgal
    Mendez-Delgado et al. 2022
    (mean of C and O gradients)
    """

    C_O_H  = 10.0**(9.12 - 0.061*Rgal  - 12.0)
    C_O_H0 = 10.0**(9.12 - 0.061*Rgal0 - 12.0)
    
    return C_O_H/C_O_H0

def get_Z_MW_Deharveng(Rgal, Rgal0=Rgal0_def):
    """Z as a function of Rgal
    Deharveng et al. (2000)
    """

    OH  = 10.0**(8.82 - 0.0394*Rgal  - 12.0)
    OH0 = 10.0**(8.82 - 0.0394*Rgal0 - 12.0)

    return OH/OH0

def get_Z_MW_AC21(Rgal, Rgal0=Rgal0_def):
    """Z as a function of Rgal
    Arellano-Cordova et al. 2021
    """

    OH  = 10.0**(8.84 - 0.042*Rgal  - 12.0)
    OH0 = 10.0**(8.84 - 0.042*Rgal0 - 12.0)
    
    return OH/OH0

def get_Z_MW_Esteban(Rgal, Rgal0=Rgal0_def):
    """Z as a function of Rgal (based on metallicity gradient)
    """

    OH0 = 10.0**(8.35 + 0.023*Rgal0 - 12.0)
    corr = OH0/10.0**(8.90 - 12.0 - 0.050*Rgal0)
    OH = np.where(Rgal >= Rgal0,
                  10.0**(8.90 - 12.0 - 0.050*Rgal)*corr,
                  10.0**(8.35 - 12.0 + 0.023*Rgal))
    
    return OH/OH0

#########################################
# Functions calculating alpha_CO and XCO
#########################################
def get_alphaCO10(Z, XCO10_0=2.0e20, muH2=muH2_def, a=0.8):
    """Compute alphaCO
    """
    try:
        Z = Z.values
    except AttributeError:
        pass

    XCO10 = XCO10_0*Z**(-a)
    alpha_CO10 = (XCO10*au.cm**-2*(au.K*au.km/au.s)**-1)*muH2*mH
    return alpha_CO10.to('Msun pc-2 K-1 km-1 s').value

def get_XCO10(Z, XCO10_0, a):
    """Compute XCO
    """
    try:
        Z = Z.values
    except AttributeError:
        pass

    XCO10 = XCO10_0*Z**(-a)*au.cm**-2*(au.K*au.km/au.s)**-1
    return XCO10.value

def get_XCO10_Gong4a(Z, a, WCO, rbeam, XCO10_0=6.1e20):
    """Compute XCO
    """
    try:
        Z = Z.values
    except AttributeError:
        pass
    try:
        WCO = WCO.values
    except AttributeError:
        pass
    try:
        rbeam = rbeam.values
    except AttributeError:
        pass

    XCO10 = XCO10_0*Z**(-a)*rbeam**(-0.25)*WCO**(-0.54+0.19*np.log10(rbeam))*\
      au.cm**-2*(au.K*au.km/au.s)**-1
    return XCO10.value


def get_alphaCO10_Lada20(Rgal, muH2=muH2_def):
    """Compute alphaCO suggested by Lada & Dame 2020
    """
    XCO10 = np.where(Rgal < 10.0,
                     83e20/(54.5 - 3.7*Rgal),
                     6.0e20)
    XCO10 *= au.cm**-2*(au.K*au.km/au.s)**-1
    return (XCO10*muH2*mH).to('Msun pc-2 K-1 km-1 s').value

###############################
# Functions calculating eps_ff
###############################
def get_epsff(avir, b):
    return np.exp(-b*avir**0.5)

def get_epsff_step(avir, avir_crit=2.0):
    return np.where(avir > avir_crit, 0.0, 1.0)

def get_SFR(M, R, avir, b=2.018, eps_cs=eps_cs_def):
    """Get SFR of a single cloud in units of Msun/yr
    """
    cl = Cloud(M, R)
    eps_ff = np.exp(-b*avir**0.5)
    return eps_cs*eps_ff*(((cl.M/cl.tff)).to('Msun yr-1').value)

def get_tff(M, R):
    """Compute free-fall time in units of Myr
    """
    cl = Cloud(M, R)
    
    return cl.tff.value


## Read MD17 data
def read_MD17_Table1(fname='./MD17/MD17-Table1.txt', force_download=False):
    """Read CO cloud catalogue of MD17
    """
    
    # Download table
    if not osp.exists(fname) or force_download:
        import bs4
        import requests

        url ='https://cfn-live-content-bucket-iop-org.s3.amazonaws.com' + \
          '/journals/0004-637X/834/1/57/1/apjaa4dfdt1_mrt.txt?' + \
          'AWSAccessKeyId=AKIAYDKQL6LTV7YY2HIK&Expires=1644988097&Signature=dO79Xbr2yryMzGFN94Ir2ymX0Z0%3D'
        
        r = requests.get(url)
        print('Downloading table from ApJ..', end='')
        assert r.ok, 'Failed to download file. Check url'
        print('Download complete..', r.elapsed)
        with open(fname, 'wb') as f:
            f.write(r.content)

    # Read original table
    df0 = pd.read_csv(fname, sep='\s+', skiprows=45,
                      names=['Cloud','Ncomp','Npix','A','l','e_l','b','e_b',
                             'theta','WCO','NH2','Sigma','vcent','sigmav','Rmax',
                             'Rmin','Rang','Rgal','INF','Dn','Df','zn','zf','Sn',
                             'Sf','Rn','Rf','Mn','Mf'])

    # Determine distance [kpc], mass [Msun], and radius [pc] based on which distance is more likely
    # INF: Gives an estimate of which distance is more likely
    #      based on the {sigma}_v_ - {Sigma}R 
    #      relation ({sigma}_v_ = 0.23({Sigma}R)^0.43{+/-}0.14_; Eq. 26).
    #     0 = Near;
    #     1 = Far.
    df0['D'] = np.where(df0['INF'] == 0, df0['Dn'], df0['Df'])
    df0['M_orig'] = np.where(df0['INF'] == 0, df0['Mn'], df0['Mf'])
    df0['R'] = np.where(df0['INF'] == 0, df0['Rn'], df0['Rf'])

    # Virial parameter (based on raw data)
    df0['avir_orig'] = 5.0*df0['sigmav']**2*df0['R']/df0['M_orig']*avir_conv
    df0['rbeam'] = rbeam_conv*df0['D']
    
    return df0


###########################

def get_SFR_all_MD17(df, a, b, XCO10_0=2.0e20, muH2=muH2_def, get_Z_function=get_Z_MW_MD22,
                     Rgal0=Rgal0_def, eps_cs=eps_cs_def, avir_cut=100, Rgal_cut=30.0, M_cut=1.0,
                     cut_before=True, verbose=False):
    """
    Function to calculate theoretical SFR_th and other quantities based on Z-dependent alphaCO
    
    a : power-law index for Z-dependent alpha_CO = alpha_{CO,0}*Z^-a
    b : coefficient in the exponential: eps_ff = eps_cs * exp(-b*avir**0.5)
    XCO10_0 : XCO10 at Z=1
    get_Z_function : function to calculate metallicity as a function of Rgal
    eps_cs : core-to-star efficiency
    cut_before : apply cut before if True
    Exclude clouds with avir > avir_cut or M < M_cut [Msun] or R > Rgal_cut [kpc]
    """

    if cut_before:
        # Apply cut first
        import copy
        df__ = copy.deepcopy(df)
        idx = df__[ (df__['avir_orig'] > avir_cut) |
                   (df__['Rgal'] > Rgal_cut) |
                   (df__['M_orig'] < M_cut) ].index
        df = df__.drop(idx, inplace=False)

        # if verbose:
        #     print('*** Sample selection ***')
        #     print('All clouds: Mtot/1e9=', df__['M_orig'].sum()/1e9)
        #     print('Selected clouds: Mtot/1e9=', df['M_orig'].sum()/1e9)
        #     print('Excluded mass (fraction)', (df__['M_orig'].sum() - df['M_orig'].sum())/df['M_orig'].sum())
        #     print('# of clouds before and after cut:', len(df__), len(df))
            
    
    mu = muH2*(1.008*au.u)

    ## Metallicity gradient and X_CO variation with Rgal
    # Compute Metallicity
    # Q: shouldn't we update Rgal information as well?

    if get_Z_function == 'Lada20':
        df['alpha_CO'] = get_alphaCO10_Lada20(df['Rgal'], muH2=muH2)
        df['XCO'] = (df['alpha_CO'].values*(au.M_sun/au.pc**2/au.K/au.km*au.s)/\
                     (muH2_def*mH)).to('cm-2 K-1 km-1 s').value
    elif get_Z_function == 'Gong4a':
        df['Z'] = get_Z_MW_MD22(df['Rgal'], Rgal0=Rgal0)
        df['XCO'] = get_XCO10_Gong4a(df['Z'], a, df['WCO']/df['Npix'], df['rbeam'])
    else:
        df['Z'] = get_Z_function(df['Rgal'], Rgal0=Rgal0)
        # alpha_CO and XCO
        df['alpha_CO'] = get_alphaCO10(df['Z'], XCO10_0=XCO10_0, muH2=muH2, a=a)
        df['XCO'] = get_XCO10(df['Z'], XCO10_0=XCO10_0, a=a)

    # Recalculate NH2m Sigma, M, and other properties using
    # Eq 14 in MD17 (note that our mu is their 2mu)
    df['NH2'] = df['WCO']*df['XCO']/df['Npix']
    df['Sigma'] = df['NH2']*(1.0*mu/au.cm**2).to('Msun pc-2').value

    # Eq 21 in MD17
    df['M'] = df['Sigma']*df['Npix']*(1e3*df['D'])**2*dOmega

    # density and free-fall time
    rho_conv = (1.0*au.M_sun/au.pc**3).to('g cm-3')
    tff_conv = ((au.cm**3/au.g/ac.G)**0.5).to('Myr')
    df['rho'] = (3.0*df['M']/(4.0*np.pi*df['R']**3))*rho_conv
    df['tff'] = ((3.0*np.pi)/(32.0*df['rho']))**0.5*tff_conv

    # Virial parameter
    df['avir'] = 5.0*df['sigmav']**2*df['R']/df['M']*avir_conv

    # SFR per freefall time
    df['epsff'] = get_epsff(df['avir'], b)
    
    # Step-function SFR per freefall time (1 if avir< 2.0; 0 otherwise)
    df['epsff_step'] = get_epsff_step(df['avir'])
    
    # Free-fall SFR [Msun/yr]
    df['SFR0'] = df['M']/df['tff']/1e6
    
    # SFR with epsff and eps_cs factors
    df['SFR'] = eps_cs*df['SFR0']*df['epsff']
    #df['SFR_step'] = eps_cs*df['SFR0']*df['epsff_step']
    df['SFR_step'] = df['SFR0']*df['epsff_step']
    
    if not cut_before:
        # Apply cut after
        import copy
        df__ = copy.deepcopy(df)
        idx = df__[ (df__['avir'] > avir_cut) |
                   (df__['Rgal'] > Rgal_cut) |
                   (df__['M'] < M_cut) ].index
        df = df__.drop(idx, inplace=False)

    if verbose:
        print('Total molecular gas mass [10^9 Msun]: ', df['M'].sum()/1e9)
        print('Total SFR_ff, SFR_th:', df['SFR0'].sum(), df['SFR'].sum())

    return df

    


def calc_MD17_SFR_grid(XCO10_0=2e20, na=200, nb=200, force_override=False):
    """Function to calculate SFR_th as functions of a and b for MD17 catalogue
    """
    import pickle
    
    fname = './MD17_SFR_grid_XCO20_{0:.1f}.p'.format(XCO10_0/1e20)
    if not force_override and osp.exists(fname):
        r = pickle.load(open(fname, 'rb'))
        return r

    df0 = read_MD17_Table1()
    a = np.linspace(-0.2,2.2,na)
    b = np.linspace(-0.2,3.2,nb)
    SFR0tot = np.zeros((na,nb))
    SFRtot = np.zeros((na,nb))
    print('na:',na, end=' ;  ')
    for i,a_ in enumerate(a):
        print(i, end=' ')
        for j,b_ in enumerate(b):
            df = get_SFR_all_MD17(df0, a=a_, b=b_, muH2=muH2_def,
                                  XCO10_0=XCO10_0, get_Z_function=get_Z_MW_MD22,
                                  cut_before=True, verbose=False)
            SFR0tot[i,j] = df['SFR0'].values.sum()
            SFRtot[i,j] = df['SFR'].values.sum()

    # Save results to pickle
    r = dict(a=a, b=b, SFR0tot=SFR0tot, SFRtot=SFRtot)


    with open(fname, 'wb') as fp:
        pickle.dump(r, fp)
    
    return r

def plt_MD17_contour_S():
    import cmocean

    r0 = calc_MD17_SFR_grid(XCO10_0=1.4e20, force_override=False)
    r1 = calc_MD17_SFR_grid(XCO10_0=2.0e20, force_override=False)
    
    fig,axes = plt.subplots(1,3,figsize=(14,6),constrained_layout=True,
                            gridspec_kw=dict(width_ratios=(0.48,0.48,0.04)))

    # cmap = plt.cm.RdBu_r
    #cmap = plt.cm.RdBu
    cmap = cmocean.cm.balance
    from matplotlib.patheffects import withStroke
    myeffect = withStroke(foreground='w', linewidth=3)

    norm = LogNorm(1/30.0,30.0)
    titles = [r'$X_{\rm CO,0}=1.4\times 10^{20}\,{\rm cm}^{-2}\,({\rm K}\,{\rm km}\,{\rm s}^{-1})^{-1}$',
              r'$X_{\rm CO,0}=2.0\times 10^{20}\,{\rm cm}^{-2}\,({\rm K}\,{\rm km}\,{\rm s}^{-1})^{-1}$']
    im = []
    lw = 2
    for i,(ax,r,title) in enumerate(zip((axes[0],axes[1]),(r0,r1),titles)):
        plt.sca(ax)
        im_ = plt.pcolormesh(r['a'], r['b'], r['SFRtot'].T/SFRtot_obs, norm=norm, cmap=cmap, shading='auto')
        ct1 = plt.contour(r['a'],r['b'],r['SFRtot'].T/SFRtot_obs,
                          [1/30.0,30.0], linestyles='--', colors='dimgrey', linewidths=lw)
        ct2 = plt.contour(r['a'],r['b'],r['SFRtot'].T/SFRtot_obs,
                          [1/10.0,10.0], linestyles='--', colors='dimgrey', linewidths=lw)
        ct3 = plt.contour(r['a'],r['b'],r['SFRtot'].T/SFRtot_obs,
                          [1/3.0,3.0], linestyles='--', colors='dimgrey', linewidths=lw)
        # Q_th=1
        ct4 = plt.contour(r['a'],r['b'],r['SFRtot'].T/SFRtot_obs,
                          [1.0], linestyles='-', colors='k', linewidths=lw+0.5)

        ax.clabel(ct4,ct4.levels,inline=True,fmt=r'$Q_{\rm th}=1$', fontsize=18,
                  manual=[(1.8,2.2)])

        plt.axhline(b_KOF21, lw=1, ls='-',c='k')
        plt.axhline(b_P12, lw=1, ls='-',c='k')
        plt.axvline(0.8, lw=1, ls='-',c='k')
        plt.axvline(1.6, lw=1, ls='-',c='k')
        plt.xlim(0,2)
        plt.ylim(0,3)
        plt.title(title)
        plt.xlabel(r'$a$', fontsize='x-large') #$\;\;(\alpha_{\rm CO} = \alpha_{\rm CO,0}Z^{-a})$')
        plt.ylabel(r'$b$', fontsize='x-large') # $\;\;(\varepsilon_{\rm ff} = \exp (-b \alpha_{\rm vir}^{1/2}))$')

        im.append(im_)

    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=axes[2], orientation='vertical', extend='neither',
                      label=r'$Q_{\rm th} \equiv {\rm SFR}_{\rm th}/{\rm SFR}_{\rm obs}$')

    fs = 20
    for ax in axes[0:-1]:
        ax.text(0.02,2.07,r'$b_{\rm KOF}$',fontsize=fs, path_effects=[myeffect])
        ax.text(0.02,1.42,r'$b_{\rm PN}$',fontsize=fs, path_effects=[myeffect])
        ax.text(0.81,0.72,r'$a_{\rm Gong}$',fontsize=fs, path_effects=[myeffect])
        ax.text(1.61,0.72,r'$a_{\rm Sun}$',fontsize=fs, path_effects=[myeffect])

    fname = osp.join(osp.expanduser('~'),
           'Dropbox/Apps/Overleaf/SFRModels (MW)/figures/fig-S-contour.png')
    plt.savefig(fname, dpi=200)

    r = dict(fig=fig, r0=r0, r1=r1, im=im)

    return r

def get_Sigma_SFR_KE12(SFR_tot_norm=SFRtot_obs):
    
    Rgal = np.array([1,3,5,7,9,11,13,15])
    log_Sigma_SFR = np.array([0.25,0.44,0.65,0.45,0.4,0.2,0.046,-0.5])
    Sigma_SFR = 10.0**log_Sigma_SFR
    
    Area = np.pi*Rgal**2
    dArea = np.zeros_like(Area)
    dArea[0] = Area[0]
    dArea[1:] = Area[1:] - Area[:-1]

    SFR_ = dArea*Sigma_SFR
    # normalize so that the total SFR matches the observed rate
    SFR = SFR_tot_norm*SFR_/SFR_.sum()
    fac = SFR_tot_norm/SFR_.sum()
    Sigma_SFR = fac*Sigma_SFR
    
    r = dict()
    r['Rgal'] = Rgal
    r['Sigma_SFR'] = Sigma_SFR
    r['dArea'] = dArea
    r['norm_fac'] = fac
    r['SFR'] = dArea*Sigma_SFR
    r['SFR_tot'] = (dArea*Sigma_SFR).sum()
    
    return r







if __name__ == '__main__':

    r0 = calc_MD17_SFR_grid(XCO10_0=1.4e20, force_override=False)
    r1 = calc_MD17_SFR_grid(XCO10_0=2.0e20, force_override=False)
    print(r0,r1)
