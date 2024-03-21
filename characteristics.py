#%% IMPORTS

from scipy import signal, fftpack
from scipy.integrate import odeint, solve_ivp, cumtrapz, RK45
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, newton, least_squares, root, minimize
from scipy import interpolate
import matplotlib.pyplot as plt
import math
import scipy as sp
import numpy as np
import csv
from numba import jit
import matplotlib_inline
import warnings


warnings.filterwarnings('ignore')
# matplotlib_inline.backend_inline.set_matplotlib_formats('svg', 'pdf')

#%% DEFAULTS

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Computer Modern Roman"]
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.markersize'] = 1

cm = 1/2.54

#%% FUNCTIONS PREAMBLE

@jit(nopython=True)
def compute_values(tc, xc, Tfc, Tsc, Qc, Ainv, Lambda, Lambdas, gamma, dtc):
    Cp = np.zeros_like(tc)
    Cn = np.zeros_like(tc)

    for j in range(1, len(tc)):
        for i in range(1, len(xc)):
            temp1 = -0.5*Lambda*(Tfc[i-1, j-1] - Tsc[i-1, j-1])*dtc + \
                    Tfc[i-1, j-1]
            temp2 = 0.5*Lambdas*(Tfc[i-1, j-1] - Tsc[i-1, j-1])*dtc + \
                    Tsc[i-1, j-1] + dtc*gamma*Qc[i-1, j-1]
                    
            Tfc[i, j] = Ainv[0, 0]*temp1 + Ainv[0, 1]*temp2
            Tsc[i-1, j] = Ainv[1, 0]*temp1 + Ainv[1, 1]*temp2

    return Cp, Cn, Tfc, Tsc


#%% PARAMETERS

'''Xi =  0.0013
kappa =  235.0374
R/L =  2.0
Rep =  13723.4655
Re =  3659590.8011
Lambda =  [36.4074 28.3169 22.5379 18.2037 12.1358]
Lambdas =  [0.0154 0.0154 0.0154 0.0154 0.0154]
beta =  0.0064
betas =  1.5
eps =  [0.25 0.3  0.35 0.4  0.5 ]
gamma =  [0.0004 0.0005 0.0007 0.0008 0.0013]'''


LL = [1, 10, 100, 1000]
BB = [1e-3, 1e-2, 0.1]

xi = 0.0013
kappa = 235.0374
eps = 0.5

# Heat source
Delta = 0.5
x0 = 0.25
delta = 0.005


for Lambda in LL:
    for beta in BB:

        # Lambda = l
        Lambdas = Lambda*xi*eps/(1-eps)
        # beta =  b
        betas = kappa*beta
        
        # gamma =  (1 - eps)*Lambdas/Lambda
        gamma = Lambdas/Lambda
        
        print('Lambda = ', np.round(Lambda, 4))
        print('Lambdas = ', np.round(Lambdas, 4))
        print('beta = ', np.round(beta, 4))
        print('betas = ', np.round(betas, 4))
        print('gamma = ', np.round(gamma, 4))
        
        #%% DNS simulation
        
        dt = 2.5e-3
        dx = 1e-2
        
        t0 = 0
        tf = 300
        xf = 1
        
        t = np.arange(t0, tf + dt, dt)
        x = np.arange(0, xf + dx, dx)
        
        Nx = len(x) - 1
        Nt = len(t) - 1
        
        # Initialization
        Tf = np.zeros((Nx+1, Nt+1))
        Ts = np.zeros((Nx+1, Nt+1))
        
        Tin = 0
        
        # BCs
        Tf[0,:] = Tin
        
        # ICs
        Tf[:,0] = 0
        Ts[:,0] = 0
        
        # Matrix terms
        p = dt/dx
        
        q = beta*dt/(dx**2)
        qs = betas*dt/(dx**2)
        
        r = Lambda*dt
        rs = Lambdas*dt
        
        # Explicit form of heat transfer term
        diagonals = [np.ones(Nx)*(1 + 2*q + r + p), np.ones(Nx)*(-q),
                     np.ones(Nx)*(-p-q)]
        
        offsets = [0, 1, -1]
        A = sp.sparse.diags(diagonals, offsets).toarray()
        A[-1,-2] = -2*q - p
        A[-1,-1] = 1 + 2*q + r + p + 2*q/p
        
        A = sp.sparse.csr_matrix(A)
        
        # Explicit form of heat transfer term
        diagonals = [np.ones(Nx + 1)*(1 + 2*qs + rs), np.ones(Nx + 1)*(-qs),
                     np.ones(Nx + 1)*(-qs)]
        
        offsets = [0, 1, -1]
        As = sp.sparse.diags(diagonals, offsets).toarray()
        As[-1,-2] = -2*qs
        As[0,1] = -2*qs
        
        As = sp.sparse.csr_matrix(As)
        
        # Boundary conditions assembly
        bc0 = np.zeros(Nx)
        bc1 = np.zeros(Nx)
        
        bc0[0] = Tin
        bc1[-1] = 1
        
        AA = sp.sparse.block_diag((sp.sparse.linalg.inv(A), sp.sparse.linalg.inv(As)))
        
        
        Q = (np.tanh(t/Delta)*np.exp(-(1/delta)*(x[:,np.newaxis] - x0)**2)/np.sqrt(np.pi*delta) +
             np.tanh(t/Delta)*np.exp(-(1/delta)*(x[:,np.newaxis] - 0.50)**2)/np.sqrt(np.pi*delta) +
             np.tanh(t/Delta)*np.exp(-(1/delta)*(x[:,np.newaxis] - 0.75)**2)/np.sqrt(np.pi*delta))/3
        
#         Q = np.tanh(t/Delta)*np.exp(-(1/delta)*(x[:,np.newaxis] -
#                                                 x0)**2)/np.sqrt(np.pi*delta)
        
        
        # PLOT
        fig, ax = plt.subplots(figsize=(7*cm,7*cm))
        
        cs1 = ax.contourf(x, t, Q.T, 16)
        for c in cs1.collections:
            c.set_rasterized(True)
            
        cbar1 = plt.colorbar(cs1)
        cbar1.set_label(r'$q$', fontsize=16)
        ax.set_xlabel(r'$\eta$', fontsize=16)
        ax.set_ylabel(r'$\tau$', fontsize=16)
        
        fig.savefig('./figs/q_vs_tau_vs_eta' + str(np.round(Lambda, 2))
                    + '_Lambdas_' + str(np.round(Lambdas, 2)) + '_beta_' +
                    str(np.round(beta, 4)) + '_betas_' + str(np.round(betas, 4)) +
                    '_x0_' + str(np.round(x0, 2)) + '_eps_' + str(np.round(eps, 2)) +
                    '.pdf', bbox_inches='tight')
        
        plt.show()
        
        
        ### TIME LOOP
        
        for i in range(1,Nt+1):
            
            bc1[-1] = 2*q/p*Tf[-1,i-1]
        #     TT = np.concatenate([Tf[1:Nx+1,i-1] + (q + p)*bc0 + r*Ts[1:,i-1] + bc1 + eps*dt*Q[1:,i-1],
        #                          Ts[:,i-1] + rs*Tf[:Nx+1,i-1] + dt*gamma*Q[:,i-1]])
        
            TT = np.concatenate([Tf[1:Nx+1,i-1] + (q + p)*bc0 + r*Ts[1:,i-1] + bc1,
                                 Ts[:,i-1] + rs*Tf[:Nx+1,i-1] + dt*gamma*Q[:,i-1]])  
            
            sol = AA.dot(TT)
            
            Tf[1:Nx+1,i] = sol[:Nx]
            Ts[:,i] = sol[Nx:]
            
        
        #
        thetaf = Tf
        thetas = Ts
        #
        
        
        #
        # PLOT
        #
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7*cm,11*cm), sharex='row')
        (ax1, ax2) = axs
        
        cs1 = ax1.contourf(x, t, thetaf.T, 16)
        for c in cs1.collections:
            c.set_rasterized(True)
            
        plt.tight_layout()
        cbar1 = plt.colorbar(cs1)
        cbar1.set_label(r'$\theta$', fontsize=16)
        ax1.set_ylabel(r'$\tau$', fontsize=16)
        
        cs2 = ax2.contourf(x, t, thetas.T, 16)
        for c in cs2.collections:
            c.set_rasterized(True)
            
        cbar2 = plt.colorbar(cs2)
        cbar2.set_label(r'$\theta_s$', fontsize=16)
        ax2.set_xlabel(r'$\eta$', fontsize=16)
        ax2.set_ylabel(r'$\tau$', fontsize=16)
        ax2.set_xlim([0, 1])
        
        fig.savefig('./figs/DNS_T_vs_x_vs_t_Lambda_' + str(np.round(Lambda, 2))
                    + '_Lambdas_' + str(np.round(Lambdas, 2)) + '_beta_' +
                    str(np.round(beta, 4)) + '_betas_' + str(np.round(betas, 4)) +
                    '_x0_' + str(np.round(x0, 2)) + '_eps_' + str(np.round(eps, 2)) +
                    '.pdf', bbox_inches='tight')
        plt.show()
        
        
        # #
        # # PLOT
        # #
        # fig, ax = plt.subplots(figsize=(7*cm,7*cm))
        
        # plt.plot(x, thetaf[:,0], 'k', x, thetas[:,0], 'k--',
        #          x, thetaf[:,int(1/4*len(t))], 
        #          x, thetaf[:,int(2/4*len(t))],
        #          x, thetaf[:,int(3/4*len(t))],
        #          x, thetaf[:,-1])
        
        # plt.gca().set_prop_cycle(None)
        
        # plt.plot(x, thetas[:,int(1/4*len(t))], '--', 
        #          x, thetas[:,int(2/4*len(t))], '--',
        #          x, thetas[:,int(3/4*len(t))], '--',
        #          x, thetas[:,-1], '--')
        
        # plt.legend([r'$t = 0$ (fluid)', r'$t = 0$ (solid)', r'$t = 1/4 \,t_f$',
        #             r'$t = 1/2 \,t_f$', r'$t = 3/4 \,t_f$', r'$t = t_f$'])
        
        # ax.set_xlabel(r'$x$', fontsize=16)
        # ax.set_ylabel(r'$T$', fontsize=16)
        
        # fig.savefig('./figs/DNS_Tf_Ts_vs_x_Lambda_' + str(np.round(Lambda, 2))
        #             + '_Lambdas_' + str(np.round(Lambdas, 2)) + '_beta_' +
        #             str(np.round(beta, 2)) + '_betas_' + str(np.round(betas, 2)) +
        #             '.pdf', bbox_inches='tight')
        # plt.show()
        
        
        
        #
        # PLOT
        #
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14*cm,7*cm), sharey='all')
        (ax1, ax2) = axs
        
        ax1.plot(t, thetaf[0,:]*0, 'k-',
                 t, thetas[0,:], 'k--',
                 t, thetaf[int(1/4*len(x)),:],
                 t, thetaf[int(1/2*len(x)),:],
                 t, thetaf[int(3/4*len(x)),:],
                 t, thetaf[-1,:])
        
        ax1.set_prop_cycle(None)
        
        ax1.plot(t, thetas[int(1/4*len(x)),:], '--',
                 t, thetas[int(1/2*len(x)),:], '--',
                 t, thetas[int(3/4*len(x)),:], '--',
                 t, thetas[-1,:], '--')
        
        ax1.legend([r'$\eta = 0$ (fluid)', r'$\eta = 0$ (solid)',
                    r'$\eta = 1/4$', r'$\eta = 1/2$', r'$\eta = 3/4$', r'$\eta = 1$'])
        #ax1.set_box_aspect(0.8)
        ax1.set_xlabel(r'$\tau$', fontsize=16)
        ax1.set_ylabel(r'$\theta$', fontsize=16)
        
        ##
        
        ax2.plot(x, thetaf[:,0], 'k',
                 x, thetas[:,0], 'k--',
                 x, thetaf[:,int(1/4*len(t))], 
                 x, thetaf[:,int(2/4*len(t))],
                 x, thetaf[:,int(3/4*len(t))],
                 x, thetaf[:,-1])
        
        plt.gca().set_prop_cycle(None)
        
        ax2.plot(x, thetas[:,int(1/4*len(t))], '--', 
                 x, thetas[:,int(2/4*len(t))], '--',
                 x, thetas[:,int(3/4*len(t))], '--',
                 x, thetas[:,-1], '--')
        
        ax2.legend([r'$\tau = 0$ (fluid)', r'$\tau = 0$ (solid)',
                    r'$\tau = 1/4 \,\tau_f$', r'$\tau = 1/2 \,\tau_f$',
                    r'$\tau = 3/4 \,\tau_f$', r'$\tau = \tau_f$'])
        # ax2.set_box_aspect(0.8)
        ax2.set_xlabel(r'$\eta$', fontsize=16)
        
        fig.savefig('./figs/DNS_T_vs_x_vs_t_subplot_Lambda_' + str(np.round(Lambda, 2))
                    + '_Lambdas_' + str(np.round(Lambdas, 2)) + '_beta_' +
                    str(np.round(beta, 4)) + '_betas_' + str(np.round(betas, 4)) +
                    '_x0_' + str(np.round(x0, 2)) + '_eps_' + str(np.round(eps, 2)) +
                    '.pdf', bbox_inches='tight')
        
        
        
        #%% Characteristics simulation
        
        CFL = dt/dx
        
        dtc = dt
        dxc = dtc
        
        xc = np.arange(0, 1 + dxc, dxc)
        tc = np.arange(0, tf + dtc, dtc)
        
        # Discretización
        xvc, tvc = np.meshgrid(xc, tc, indexing='ij')
        
        Cp = xvc*0
        Cn = xvc*0
        
        Tsc = xvc*0
        Tfc = xvc*0
        
        # BCs
        Tfc[0,:] = Tin
        
        # ICs
        Tfc[:,0] = 0
        Tsc[:,0] = 0
        
        # Valor inicial de las características
        Cp[:,0] = tc[0] + xc
        Cn[:,0] = xc
        
        A = [[1 + 0.5*Lambda*dtc, -0.5*Lambda*dtc],
            [-0.5*Lambdas*dtc, 1 + 0.5*Lambdas*dtc]]
        Ainv = np.linalg.inv(A)
        
        # Heat source with characteristics discretization
#         Qc = np.tanh(tc/Delta)*np.exp(-(1/delta)*(xc[:,np.newaxis] - x0)**2)/np.sqrt(np.pi*delta)

        Qc = (np.tanh(tc/Delta)*np.exp(-(1/delta)*(xc[:,np.newaxis] - x0)**2)/np.sqrt(np.pi*delta) +
              np.tanh(tc/Delta)*np.exp(-(1/delta)*(xc[:,np.newaxis] - 0.50)**2)/np.sqrt(np.pi*delta) +
              np.tanh(tc/Delta)*np.exp(-(1/delta)*(xc[:,np.newaxis] - 0.75)**2)/np.sqrt(np.pi*delta))/3


        # Numerical solution
        Cp, Cn, Tfc, Tsc = compute_values(tc, xc, Tfc, Tsc, Qc, Ainv,
                                          Lambda, Lambdas, gamma, dtc)
        
        
        #
        # PLOT
        #
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7*cm,11*cm), sharex='row')
        (ax1, ax2) = axs
        
        cs1 = ax1.contourf(xvc, tvc, Tfc, 16)
        for c in cs1.collections:
            c.set_rasterized(True)
            
        plt.tight_layout()
        cbar1 = plt.colorbar(cs1)
        cbar1.set_label(r'$\theta$', fontsize=16)
        ax1.set_ylabel(r'$\tau$', fontsize=16)
        # ax1.plot(xc, xc,'r--', label=r'$x = t$')
        
        
        cs2 = ax2.contourf(xvc, tvc, Tsc, 16)
        for c in cs2.collections:
            c.set_rasterized(True)
            
        cbar2 = plt.colorbar(cs2)
        cbar2.set_label(r'$\theta_s$', fontsize=16)
        ax2.set_ylabel(r'$\tau$', fontsize=16)
        # ax2.set_box_aspect(1)
        # ax2.plot(xc, xc,'r--', label=r'$x = t$')
        ax2.set_xlim([0, 1])
        #ax2.set_ylim([0, 1])
        ax2.set_xlabel(r'$\eta$', fontsize=16)
        
        fig.savefig('./figs/char_T_vs_x_vs_t_Lambda_' + str(np.round(Lambda, 2))
                    + '_Lambdas_' + str(np.round(Lambdas, 2)) + '_beta_' +
                    str(np.round(beta, 4)) + '_betas_' + str(np.round(betas, 4)) +
                    '_x0_' + str(np.round(x0, 2)) + '_eps_' + str(np.round(eps, 2)) +
                    '.pdf', bbox_inches='tight')
        
        
        #
        # PLOT
        #
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14*cm,7*cm), sharey='all')
        (ax1, ax2) = axs
        
        ax1.plot(tc, Tfc[0,:]*0, 'k-',
                 tc, Tsc[0,:], 'k--',
                 tc, Tfc[int(1/4*len(xc)),:],
                 tc, Tfc[int(1/2*len(xc)),:],
                 tc, Tfc[int(3/4*len(xc)),:],
                 tc, Tfc[-1,:])
        
        ax1.set_prop_cycle(None)
        
        ax1.plot(tc, Tsc[int(1/4*len(xc)),:], '--',
                 tc, Tsc[int(1/2*len(xc)),:], '--',
                 tc, Tsc[int(3/4*len(xc)),:], '--',
                 tc, Tsc[-1,:], '--')
        
        ax1.legend([r'$\eta = 0$ (fluid)', r'$\eta = 0$ (solid)',
                    r'$\eta = 1/4$', r'$\eta = 1/2$', r'$\eta = 3/4$', r'$\eta = 1$'])
        ax1.set_xlabel(r'$\tau$', fontsize=16)
        ax1.set_ylabel(r'$\theta$', fontsize=16)
        
        ##
        
        ax2.plot(xc, Tfc[:,0], 'k',
                 xc, Tfc[:,0], 'k--',
                 xc, Tfc[:,int(1/4*len(tc))], 
                 xc, Tfc[:,int(2/4*len(tc))],
                 xc, Tfc[:,int(3/4*len(tc))],
                 xc, Tfc[:,-1])
        
        plt.gca().set_prop_cycle(None)
        
        ax2.plot(xc, Tsc[:,int(1/4*len(tc))], '--', 
                 xc, Tsc[:,int(2/4*len(tc))], '--',
                 xc, Tsc[:,int(3/4*len(tc))], '--',
                 xc, Tsc[:,-1], '--')
        
        ax2.legend([r'$\tau = 0$ (fluid)', r'$\tau = 0$ (solid)',
                    r'$\tau = 1/4 \,\tau_f$', r'$\tau = 1/2 \,\tau_f$',
                    r'$\tau = 3/4 \,\tau_f$', r'$\tau = \tau_f$'])
        ax2.set_xlabel(r'$\eta$', fontsize=16)
        
        # fig.savefig('./figs/DNS_T_vs_x_vs_t_subplot_Lambda_' + str(np.round(Lambda, 2))
        #             + '_Lambdas_' + str(np.round(Lambdas, 2)) +
        #             '_beta_' + str(np.round(beta, 2)) + '_betas_' +
        #             str(np.round(betas, 2)) + '.pdf', bbox_inches='tight')
        
        
        
        
        #
        # PLOT
        #
        fig, ax = plt.subplots(figsize=(9*cm,9*cm))
        
        # DNS - fluid
        plt.plot(x, thetaf[:,0], 'b-')
        plt.plot(xc, Tfc[:,0], 'b--')
        plt.plot(x, thetas[:,0], 'r-')
        plt.plot(xc[:-1], Tsc[:-1,0], 'r--')
        
        plt.plot(x, thetaf[:,int(0.25/dt)], 'b-', label='DNS-fluid')
        plt.plot(x, thetaf[:,int(0.50/dt)], 'b-')
        plt.plot(x, thetaf[:,int(1/4*len(t))], 'b-')
        plt.plot(x, thetaf[:,int(2/4*len(t))], 'b-')
        plt.plot(x, thetaf[:,int(3/4*len(t))], 'b-')
        plt.plot(x, thetaf[:,-1], 'b-')
        
        # DNS - solid
        plt.plot(x, thetas[:,int(0.25/dt)], 'r-')
        plt.plot(x, thetas[:,int(0.50/dt)], 'r-')
        plt.plot(x, thetas[:,int(1/4*len(t))], 'r-', label='DNS-solid')
        plt.plot(x, thetas[:,int(2/4*len(t))], 'r-')
        plt.plot(x, thetas[:,int(3/4*len(t))], 'r-')
        plt.plot(x, thetas[:,-1], 'r-')
        
        # Char. - fluid
        plt.plot(xc[:-1], Tfc[:-1,int(0.25/dtc)], 'b--')
        plt.plot(xc[:-1], Tfc[:-1,int(0.50/dtc)], 'b--')
        plt.plot(xc[:-1], Tfc[:-1,int(1/4*len(tc))], 'b--', label='Char.-fluid')
        plt.plot(xc[:-1], Tfc[:-1,int(2/4*len(tc))], 'b--')
        plt.plot(xc[:-1], Tfc[:-1,int(3/4*len(tc))], 'b--')
        plt.plot(xc[:-1], Tfc[:-1,-1], 'b--')
        
        # Char. - solid
        plt.plot(xc[:-1], Tsc[:-1,int(0.25/dtc)], 'r--')
        plt.plot(xc[:-1], Tsc[:-1,int(0.50/dtc)], 'r--')
        plt.plot(xc[:-1], Tsc[:-1,int(1/4*len(tc))], 'r--', label='Char.-solid')
        plt.plot(xc[:-1], Tsc[:-1,int(2/4*len(tc))], 'r--')
        plt.plot(xc[:-1], Tsc[:-1,int(3/4*len(tc))], 'r--')
        plt.plot(xc[:-1], Tsc[:-1,-1], 'r--')
        
        # Average T
        # plt.plot(xc + 0.05, xc*0 + 0.5, 'k-.')
        
        '''
        # Cosmetics
        plt.text(0.83, 0.53, r'$T_{in}/2$', fontsize=14)
        plt.arrow(-0.05, 0.85, 1, -0.5, width=0.0001, color='k', head_width=0.0125, head_length=0.025, lw=0.5)
        plt.text(0.80, 0.25, r'$t = 1/4,\, 1/2,\,1/4\,t_f,$', fontsize=12) 
        plt.text(0.865, 0.20, r'$1/2\,t_f,3/4\,t_f,\,t_f.$', fontsize=12)
        '''
        
        plt.xlim([-0.125, xc[-1] + 0.125])
        
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        ax.set_box_aspect(0.8)
        ax.set_xlabel(r'$\eta$', fontsize=16)
        ax.set_ylabel(r'$\theta$', fontsize=16)
        
        fig.savefig('./figs/DNS_vs_characteristics_all_Lambda_' + str(np.round(Lambda, 2))
                    + '_Lambdas_' + str(np.round(Lambdas, 2)) + '_beta_' +
                    str(np.round(beta, 4)) + '_betas_' + str(np.round(betas, 4)) +
                    '_x0_' + str(np.round(x0, 2)) + '_eps_' + str(np.round(eps, 2)) +
                    '.pdf', bbox_inches='tight')
        plt.show()

