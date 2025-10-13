kinetic_path = "/home/katanovic/fluid_kinetic_class/"
phare_path = "/home/katanovic/PHARE/"

import os
import sys

phare_root = os.path.expanduser(phare_path)
sys.path.append(os.path.join(phare_path, "pyphare"))
sys.path.append(os.path.join(kinetic_path, "2024"))

import matplotlib.pyplot as plt
import numpy as np
from numpy import polyfit
from pyphare.pharesee.hierarchy import fromh5  # was get_times_from_h5
from pyphare.pharesee.run import Run
from scipy.ndimage import gaussian_filter1d as gf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy as sc
from scripts import dist_plot

root_path = "/home/katanovic/fluid_kinetic_class/2024/run/beam/"

from get_get import *

def plot_Exyz(r, times, sigma=2):
    ex, ey, ez, x = get_E(r, times, sigma)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    for it, t in enumerate(times):
        ax.plot(x, ex[:,it], label=fr"$E_x$ at $t={t}$")
        ax.plot(x, ey[:,it], label=fr"$E_y$ at $t={t}$")
        ax.plot(x, ez[:,it], label=fr"$E_z$ at $t={t}$")

    ax.set_title("Electric field components $E_x$, $E_y$, $E_z$ for various times (linear and non-linear part)")
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude$")
    ax.legend(loc='upper right', fontsize='small', ncol=2)

def plot_densities(r, times, sigma=2):
    x2, m = get_densities(r, times, sigma)
    fig, ax = plt.subplots(1,3, figsize=(14,4), constrained_layout=True)

    fig.suptitle("Wave propagation for 3 timelines")
    
    for i in range(0, 3):
        ax[0].plot(x2, m[:,i], label=f"t = {times[i]}")
    ax[0].legend()
    ax[0].set_title("Before mode n°6 diminantes")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("Density")
    
    for i in range(3, 6):
        ax[1].plot(x2, m[:,i], label=f"t = {times[i]}")
    ax[1].legend()
    ax[1].set_title("Mode n°6 start to dominate")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Density")
    
    for i in range(6,9):
        ax[2].plot(x2, m[:,i], label=f"t = {times[i]}")
    ax[2].legend()
    ax[2].set_title("Mode n°6 dominates completly")
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("Density")
    return m, x2

def plot_v_phase(r, times, sigma=3):
    x, m = get_densities(r, times, sigma)
    file = os.path.join(r.path, "EM_B.h5")
    T = fromh5.get_times_from_h5(file)
    dt = T[1]-T[0]
    p1 = find_peaks(m[:,0])[0][0]
    p2 = find_peaks(m[:,1])[0][0]
    dx1 = x[p2]-x[p1]
    v1 = dx1/dt
    k = 2*np.pi*6/r.GetDomainSize()[0] #6 because 6th mode
    omega = k*v1

    fig, ax = plt.subplots(1,1, figsize=(6,4), constrained_layout=True)
    ax.plot(x, m[:,0], label=f"t = {times[0]}")
    ax.plot(x, m[:,1], label=f"t = {times[1]}")
    ax.legend()
    ax.scatter(x[p1], m[p1,0], c="black")
    ax.scatter(x[p2], m[p2,1], c="black")
    ax.axvline(x[p1], 0, c="r", ls="--")
    ax.axvline(x[p2], 0, c="r", ls="--")
    ax.set_title(f"Phase velocity = {v1:.3f}, omega = {omega:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    
    return omega, v1

def plot_ec_ue(r, times, sigma = 4):
    vx, vy, vz, x = get_velocities(r, times, sigma)
    x1, n = get_densities(r, times, sigma)
    ue, x2 = get_ue(r, times, sigma)
    Ec = 0.5*n*(vx**2+vy**2+vz**2)
    
    fig, ax = plt.subplots(1,2, figsize=(14, 4))

    for i in range(times.size):
        ax[0].plot(x1, Ec[:,i], label=f"time = {times[i]}")
    
    for i in range(times.size):
        ax[1].plot(x2, ue[:,i], label=f"time = {times[i]}")
    
    ax[0].legend()
    ax[0].set_title("Kinetic energy")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("Ec")
    ax[1].legend()
    ax[1].set_title("Electromagnetic energy")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("Ue")
    return Ec, ue

def plot_pressure(r, times, sigma = 0):
    n = len(times)
    fig, ax = plt.subplots(n ,1, figsize=(6, 4*n), constrained_layout=True)
    
    if n == 1:
        ax = [ax]
    pxx, pxy, pxz, pyy, pyz, pzz, x = get_pressures(r, times, sigma)
    for i in range(n):
        A = np.average((pyy[:,i] + pzz[:,i]) / (2 * pxx[:,i]))
        if A != 1:
            type_ = "cisaillement" if A > 1 else "compression"
            title = f"t = {times[i]} — Pression anisotropique de type {type_} A = {A:.2f}"
        else:
            title = f"t = {times[i]} — Pression isotropique A = {A:.2f}"
        ax[i].set_title(title)
        ax[i].plot(x, pxx[:,i], label="pxx")
        ax[i].plot(x, pyy[:,i], label="pyy")
        ax[i].plot(x, pzz[:,i], label="pzz")
        ax[i].set_xlabel("x")
        ax[i].set_ylabel("Intensity")
        ax[i].legend()

def plot_polar_byz_eyz(r, times, sigma=1):
    bx, by, bz, x = get_B(r, times, sigma)
    ex, ey, ez, x = get_E(r, times, sigma)
    if bx.size != ex.size:
        x = x[:-1]
        ex = ex[:-1]
        ey = ey[:-1]
        ez = ez[:-1]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    for i in range(times.size):
        ax[0, 0].plot(x, by[:,i], label=f"$B_y$ t= {times[i]}")
        ax[0, 0].plot(x, bz[:,i], label=f"$B_z$ t= {times[i]}")
    ax[0, 0].set_ylabel("Components of $\\vec{B}$")
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_title("$B_y(x)$ and $B_z(x)$")
    ax[0, 0].legend()

    for i in range(times.size):
        ax[0, 1].plot(by[:,i], bz[:,i], label=f"$B_yz$ t= {times[i]}")
        ax[0, 1].plot(by[0,i], bz[0,i], "r.", label=f"Start t= {times[i]}")
        ax[0, 1].plot(by[10,i], bz[10,i], "g.", label=f"t= {times[i]}")
    ax[0, 1].set_xlabel("$B_y$")
    ax[0, 1].set_ylabel("$B_z$")
    ax[0, 1].set_title("Evolution of $\\vec{B}$ in the y–z plane")
    ax[0, 1].set_aspect("equal", "box")
    ax[0, 1].legend()

    for i in range(times.size):
        ax[1, 0].plot(x, ey[:,i], label=f"$E_y$ t= {times[i]}")
        ax[1, 0].plot(x, ez[:,i], label=f"$E_z$ t= {times[i]}")
    ax[1, 0].set_ylabel("Components of $\\vec{E}$")
    ax[1, 0].set_xlabel("x")
    ax[1, 0].set_title("$E_y(x)$ and $E_z(x)$")
    ax[1, 0].legend()

    for i in range(times.size):
        ax[1, 1].plot(ey[:,i], ez[:,i], label=f"$E_yz$ t= {times[i]}")
        ax[1, 1].plot(ey[0, i], ez[0, i], "r.", label=f"Start t= {times[i]}")
        ax[1, 1].plot(ey[10, i], ez[10, i], "g.", label=f"t= {times[i]}")
    ax[1, 1].set_xlabel("$E_y$")
    ax[1, 1].set_ylabel("$E_z$")
    ax[1, 1].set_title("Evolution of $\\vec{E}$ in the y–z plane")
    ax[1, 1].set_aspect("equal", "box")
    ax[1, 1].legend()
    
    plt.suptitle("Visualization of Electric and Magnetic Fields $\\vec{E}$ and $\\vec{B}$", fontsize=16)


def plot_polar_b(r, times, sigma=1):
    fig1, ax1 = plt.subplots(1, 1, constrained_layout=True)
    bx, by, bz, x = get_B(r, times,sigma)
    for i in range(times.size):
        ax1.plot(by[:,i], bz[:,i], label=f"t = {times[i]}")
        ax1.legend()
    ax1.set_xlabel("$B_y$")
    ax1.set_ylabel("$B_z$")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Evolution of Magnetic Field Components Over Time")

def plot_ampli_gamma(r, number):
    file = os.path.join(r.path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)
    dt = times[1] - times[0]
    amplitude = []
    g = []
    time_shape = []
    number = 6
    for i in range(number):
        a, b, times = gamma_B(r.path,i)
        amplitude.append(a)
        g.append(b)
        time_shape.append(len(times))
        k = np.arange(1, number + 1, 1)
    k = k*2*np.pi/r.GetDomainSize()[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    
    ax[0].plot(k,amplitude,color="r",linestyle="-",label="amplitude")
    ax[0].set_title(r"Amplitude rate $\alpha$")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel(r"$\alpha$ (k)")
    ax[0].legend()
    
    ax[1].plot(k,g, color='b', linestyle='dotted', label="gamma")
    ax[1].set_title(r"Growing rate $\gamma$")
    ax[1].set_xlabel("k")
    return amplitude, g

def  plot_comparative_mode(r, amplitude1, amplitude2, g1, g2):
    file = os.path.join(r.path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)
    dt = times[1]-times[0]
    
    t = np.arange(0, np.max(times)*1.75, dt)
    one = np.ones(t.size)
    
    R1 = amplitude1*np.exp(g1*t)
    R2 = amplitude2*np.exp(g2*t)
    R = R1/R2
    
    t_cara = np.log(amplitude2/amplitude1)/(g1-g2)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    ax1.set_title(f"Ratio R = [{amplitude1:.3f}*exp({g1:.3f}*t)]/[{amplitude1:.3f}*exp({g2:.3f}*t)]")
    ax1.plot(t,R,color="r",linestyle="-", label="Ration R = R1/R2")
    ax1.plot(t,one ,color="b",linestyle="dotted", label="=1")
    ax1.axvline(t_cara, 0, color="black", linestyle="--", label=f"characteristic time = {t_cara:.2f}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Ratio")
    ax1.legend()