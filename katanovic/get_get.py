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

def get_B(r, times, sigma = 0):
    for it, t in enumerate(times):
        B = r.GetB(t, merged=True)
        bx_, x_ = B["Bx"]
        by_, x_ = B["By"]
        bz_, x_ = B["Bz"]
        x = x_[0]
        bx = bx_(x)
        by = by_(x)
        bz = bz_(x)
        if sigma > 0:
            bx = gf(bx, sigma)
            by = gf(by, sigma)
            bz = gf(bz, sigma)
        if it == 0:
            btx = np.zeros((len(bx), len(times)))
            bty = np.zeros((len(by), len(times)))
            btz = np.zeros((len(bz), len(times)))
        btx[:, it] = bx
        bty[:, it] = by
        btz[:, it] = bz
    return btx, bty, btz, x

def get_E(r, times, sigma = 0):
    for it, t in enumerate(times):
        E = r.GetE(t, merged = True, interp="linear")
        ex_, x_ = E["Ex"]
        ey_, x_ = E["Ey"]
        ez_, x_ = E["Ez"]
        x = x_[0][:-1]
        ex = ex_(x)
        ey = ey_(x)
        ez = ez_(x)
        if sigma > 0:
            ex = gf(ex, sigma)
            ey = gf(ey, sigma)
            ez = gf(ez, sigma)
        if it == 0:
            etx = np.zeros((len(ex), len(times)))
            ety = np.zeros((len(ey), len(times)))
            etz = np.zeros((len(ez), len(times)))
        etx[:, it] = ex
        ety[:, it] = ey
        etz[:, it] = ez
    return etx, ety, etz, x

def get_velocities(r, times, sigma=0):
    for it, t in enumerate(times):
        Vi = r.GetVi(t, merged=True)
        Vix = Vi["Vx"][0]
        Viy = Vi["Vy"][0]
        Viz = Vi["Vz"][0]
        xV = Vi["Vx"][1][0]
        if sigma > 0:
            vx = gf(Vix(xV), sigma)
            vy = gf(Viy(xV),sigma)
            vz = gf(Viz(xV), sigma)
        else:
            vx = Vix(xV)
            vy = Viy(xV)
            vz = Viz(xV)
            
        if it == 0:
            vtx = np.zeros((len(vx), len(times)))
            vty = np.zeros((len(vy), len(times)))
            vtz = np.zeros((len(vz), len(times)))
        vtx[:, it] = vx
        vty[:, it] = vy
        vtz[:, it] = vz
        
    return vtx, vty, vtz, xV

def get_poynting(r, times, sigma = 0):
    ex, ey, ez, x = get_E(r, times, sigma)
    bx, by, bz, x = get_B(r, times, sigma)

    sx = ey[:-1,:]*bz-ez[:-1,:]*by
    sy = ez[:-1,:]*bx-ex[:-1,:]*bz
    sz = ex[:-1,:]*by-ey[:-1,:]*bz

    S = np.sqrt(sx**2+sy**2+sz**2)
    return S, sx, sy, sz, x

def get_ue(r, times, sigma = 0):
    ex, ey, ez, x = get_E(r, times, sigma)
    bx, by, bz, x = get_B(r, times, sigma)

    ue = 0.5*(sum((ex**2+ey**2+ez**2),
                  (bx**2+by**2+bz**2)))
    return ue, x

def get_J(r, times, sigma = 0):
    for it, t in enumerate(times):
        J = r.GetJ(t, merged=True, interp="nearest")
        jy, x_ = J["Jy"]
        jz, x_ = J["Jz"]
        x = x_[0]
        if sigma > 0:
            jy = gf(jy(x), 2)
            jz = gf(jz(x), 2)
        if it == 0:
            jty = np.zeros((len(jy), len(times)))
            jtz = np.zeros((len(jz), len(times)))
        jty[:, it] = jy
        jtz[:, it] = jz
    return jty, jtz, x

def get_densities(r,times, sigma=0):
    for it,t in enumerate(times):
        N = r.GetNi(t, merged=True)
        Ni = N["rho"][0]
        x = N["rho"][1][0]
        if sigma >0:
            n = gf(Ni(x),sigma)
        else:
            n = Ni(x)
        if it==0 :
            nt = np.zeros((len(n),len(times)))
        nt[:,it] = n
    return x, nt

def get_potentiel(r, times, sigma = 0):
    for it, t in enumerate(times):
        E = r.GetE(t, merged=True)
        ex_, x_ = E["Ex"]
        ey_, x_ = E["Ey"]
        ez_, x_ = E["Ez"]
        x = x_[0]
        ex = gf(ex_(x), sigma)
        ey = gf(ey_(x), sigma)
        ez = gf(ez_(x), sigma)
        dex = np.zeros(ex.size)
        dey = np.zeros(ex.size)
        dez = np.zeros(ex.size)
        for i in range(ex.size - 1):
            dex[i] = -(ex[i + 1] - (ex[i])) / (x[i + 1] - x[i])
            dey[i] = -(ey[i + 1] - (ey[i])) / (x[i + 1] - x[i])
            dez[i] = -(ez[i + 1] - (ez[i])) / (x[i + 1] - x[i])
        if it == 0:
            dtx = np.zeros((len(dex), len(times)))
            dty = np.zeros((len(dey), len(times)))
            dtz = np.zeros((len(dey), len(times)))
        dtx[:,it] = dex
        dty[:,it] = dey
        dtz[:,it] = dez
    return dtx, dty, dtz, x

def get_pressures(r, times, sigma=0):
    for it, t in enumerate(times):
        M = r.GetM(t, merged=True)
        massDensity = r.GetMassDensity(t, merged=True)
        V = r.GetVi(t, merged=True)
        Mxx = M["Mxx"][0]
        Mxy = M["Mxy"][0]
        Mxz = M["Mxz"][0]
        Myy = M["Myy"][0]
        Myz = M["Myz"][0]
        Mzz = M["Mzz"][0]
        x = M["Mxx"][1][0]
        md = massDensity["rho"][0]
        Vix = V["Vx"][0]
        Viy = V["Vy"][0]
        Viz = V["Vz"][0]
        Pxx = Mxx(x) - Vix(x) * Vix(x) * md(x)
        Pxy = Mxy(x) - Vix(x) * Viy(x) * md(x)
        Pxz = Mxz(x) - Vix(x) * Viz(x) * md(x)
        Pyy = Myy(x) - Viy(x) * Viy(x) * md(x)
        Pyz = Myz(x) - Viy(x) * Viz(x) * md(x)
        Pzz = Mzz(x) - Viz(x) * Viz(x) * md(x)
        pxx = gf(Pxx, sigma)
        pxy = gf(Pxy, sigma)
        pxz = gf(Pxz, sigma)
        pyy = gf(Pyy, sigma)
        pyz = gf(Pyz, sigma)
        pzz = gf(Pzz, sigma)
        if it == 0:
            pxxt = np.zeros((len(pxx), len(times)))
            pxyt = np.zeros((len(pxx), len(times)))
            pxzt = np.zeros((len(pxx), len(times)))
            pyyt = np.zeros((len(pxx), len(times)))
            pyzt = np.zeros((len(pxx), len(times)))
            pzzt = np.zeros((len(pxx), len(times)))
        pxxt[:,it] = pxx
        pxyt[:,it] = pxy
        pxzt[:,it] = pxz
        pyyt[:,it] = pyy
        pyzt[:,it] = pyz
        pzzt[:,it] = pzz
    return pxxt, pxyt, pxzt, pyyt, pyzt, pzzt, x

def get_massDensity(r,times, sigma=0):
    for it,t in enumerate(times):
        Md = r.GetMassDensity(t, merged=True)
        rM = Md["rho"][0]
        xM = Md["rho"][1][0]
        if sigma >0:
            m = gf(rM(xM),sigma)
        else:
            m = rM(xM)
        if it==0 :
            mt = np.zeros((len(m),len(times)))
        mt[:,it] = m
    return xM, mt

def fourier_series(r, time, mode, field, direction=None):
    if field == "E":
        ex, ey, ez, x = get_E(r, time, sigma=0)
        if direction == "x":
            data = ex
        elif direction == "y":
            data = ey
        elif direction == "z":
            data = ez

    elif field == "B":
        bx, by, bz, x = get_B(r, time, sigma=0)
        if direction == "x":
            data = bx
        elif direction == "y":
            data = by
        elif direction == "z":
            data = bz
            
    elif field == "V":
        vx, vy, vz, x = get_velocities(r, time, sigma=0)
        if direction == "x":
            data = vx
        elif direction == "y":
            data = vy
        elif direction == "z":
            data = vz
            
    elif field == "N":
        x, n = get_densities(r, time, sigma=0)
        data = n

    e_k = np.fft.fft(data, axis=0)

    e_k_mode = np.zeros_like(e_k, dtype=complex)
    e_k_mode[mode, :] = e_k[mode, :]
    e_k_mode[-mode, :] = e_k[-mode, :]

    fs = np.fft.ifft(e_k_mode, axis=0)
    energie_mode = np.sum(fs.real)
    return fs, x, energie_mode

def growth_b_right_hand(run_path, time_offset, i):
    file = os.path.join(run_path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)  # Was get_times_from_h5 #Was none
    dt = times[1] - times[0]
    r = Run(run_path)
    first_mode = np.array([])

    for time in times:
        B_hier = r.GetB(time, merged=True, interp="linear")

        by_interpolator, xyz_finest = B_hier["By"]
        bz_interpolator, xyz_finest = B_hier["Bz"]
        
        # remove the last point so that "x" is periodic wo. last point = first point
        x = xyz_finest[0][:-1]

        by = by_interpolator(x)
        bz = bz_interpolator(x)

        # get the mode 1, as it is the most unstable in a box of length 33
        mode1 = np.absolute(np.fft.fft(by - 1j * bz)[i+1])
        first_mode = np.append(first_mode, mode1)

    ioffset = int(time_offset / dt)
    imax = find_peaks(first_mode, width=ioffset)[0][0] #Without width=ioffset
    # the curve_fit is performed from time index 0 to imax-ioffset as this offset prevent to use
    # the final part of the curve which is no more exponential as this is the end of the linear mode
    popt, pcov = curve_fit(
        yaebx, times[: imax - ioffset], first_mode[: imax - ioffset], p0=[0.08, 0.09]
    )

    # now the signal is stripped from its exponential part
    damped_mode = first_mode[: imax - ioffset] * yaebx(
        times[: imax - ioffset], 1 / popt[0], -popt[1]
    )

    # find the omega for which "damped_mode" is the largest :
    # this term is twice the one it should be because "mode1" resulting from
    # an absolute value, this (cosx)^2 = cos(2x) then appears at the 2nd
    # harmonoic (hence the factor 0.5 to get "omega")
    # the factor "+1" is because we remove the DC component, so the value
    # given by argmax has also to miss this value
    omegas = np.fabs(np.fft.fft(damped_mode).real)
    omega = (
        0.5
        * (omegas[1 : (omegas.size // 2)+1].argmax() + 1)
        * 2
        * np.pi
        / times[imax - 1 - ioffset]
    )

    return times, first_mode, popt[0], popt[1], damped_mode, omega

def growth_b_left_hand(run_path, time_offset, i):
    file = os.path.join(run_path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)  # Was get_times_from_h5 #Was none
    dt = times[1] - times[0]
    r = Run(run_path)
    first_mode = np.array([])

    for time in times:
        B_hier = r.GetB(time, merged=True, interp="linear")

        by_interpolator, xyz_finest = B_hier["By"]
        bz_interpolator, xyz_finest = B_hier["Bz"]
        
        # remove the last point so that "x" is periodic wo. last point = first point
        x = xyz_finest[0][:-1]

        by = by_interpolator(x)
        bz = bz_interpolator(x)

        # get the mode 1, as it is the most unstable in a box of length 33
        mode1 = np.absolute(np.fft.fft(by + 1j * bz)[i+1])
        first_mode = np.append(first_mode, mode1)

    ioffset = int(time_offset / dt)
    imax = find_peaks(first_mode, width=ioffset)[0][0] #Without width=ioffset
    # the curve_fit is performed from time index 0 to imax-ioffset as this offset prevent to use
    # the final part of the curve which is no more exponential as this is the end of the linear mode
    popt, pcov = curve_fit(
        yaebx, times[: imax - ioffset], first_mode[: imax - ioffset], p0=[0.08, 0.09]
    )

    # now the signal is stripped from its exponential part
    damped_mode = first_mode[: imax - ioffset] * yaebx(
        times[: imax - ioffset], 1 / popt[0], -popt[1]
    )

    # find the omega for which "damped_mode" is the largest :
    # this term is twice the one it should be because "mode1" resulting from
    # an absolute value, this (cosx)^2 = cos(2x) then appears at the 2nd
    # harmonoic (hence the factor 0.5 to get "omega")
    # the factor "+1" is because we remove the DC component, so the value
    # given by argmax has also to miss this value
    omegas = np.fabs(np.fft.fft(damped_mode).real)
    omega = (
        0.5
        * (omegas[1 : (omegas.size // 2)+1].argmax() + 1)
        * 2
        * np.pi
        / times[imax - 1 - ioffset]
    )

    return times, first_mode, popt[0], popt[1], damped_mode, omega

def all_distrib(r, time):
    main = r.GetParticles(time, ["main"])
    beam = r.GetParticles(time, ["beam"])
    both = r.GetParticles(time, ["main", "beam"])
    #md = dist_plot(main, axis=("x","Vx"), title=f"Main distribution at t = {time}")
    #bd = dist_plot(beam, axis=("x","Vx"), title =f"Beam distribution at t = {time}")
    #alld = dist_plot(both, axis=("x", "Vx"), title=f"Main + Beam distribution at t = {time}")
    #Vd = dist_plot(both, axis=("Vx", "Vy"), title=f"Velocity distribution at t = {time}")

    #sp_all = dist_plot(main, axis=("x", "Vx"), plot_type="scatter", title="Particule distribution")
    #sp_all = dist_plot(beam, axis=("x", "Vx"), plot_type="scatter", title="Particule distribution")
    sp_all = dist_plot(both, axis=("x", "Vx"), plot_type="scatter", title="Particule distribution")
    return sp_all

def compare_dist_plots(r, particles, title1="Distribution for t = 0", title2="Distribution for t = 99", **kwargs):
    """
    Affiche côte à côte deux distributions de particules en phase space.
    
    kwargs sont transmis à dist_plot pour configurer les deux plots.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    kwargs1 = kwargs.copy()
    kwargs2 = kwargs.copy()
    kwargs1["ax"] = ax1
    kwargs1["title"] = title1
    kwargs2["ax"] = ax2
    kwargs2["title"] = title2

    if particles == "main":
        particles1 = r.GetParticles(0, ["main"])
        particles2 = r.GetParticles(99, ["main"])
    elif particles == "beam":
        particles1 = r.GetParticles(0, ["beam"])
        particles2 = r.GetParticles(99, ["beam"])
    elif particles == "both":
        particles1 = r.GetParticles(0, ["main", "beam"])
        particles2 = r.GetParticles(99, ["main", "beam"])
    else:
        raise ValueError("particles must be 'main', 'beam', or 'both'")

    if "scatter" in kwargs:
        kwargs1["ax"] = ("x", "Vx")
        kwargs2["ax"] = ("x", "Vx")

    dist_plot(particles1, **kwargs1)
    dist_plot(particles2, **kwargs2)
    
    return fig, (ax1, ax2)

def yaebx(x, a, b):
    return a * np.exp(np.multiply(b, x))

def yax(x, a, b):
    return a*x+b

def phase_space_ex(time, r, **kwargs):
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)

    vmin, vmax = kwargs.get("vmin", -1.2), kwargs.get("vmin", 7.6)

    ions = r.GetParticles(time, ["main", "beam"])

    ions.dist_plot(
        axis=("x", "Vx"),
        ax=ax1,
        norm=0.4,
        finest=True,
        gaussian_filter_sigma=(1, 1),
        vmin=vmin,
        vmax=vmax,
        dv=0.05,
        title="t = {:.1f}".format(time),
        xlabel="",
        ylabel="",
    )
    ax2 = ax1.twinx()

    E_hier = r.GetE(time, merged=True, interp="linear")
    ex_interpolator, xyz_finest = E_hier["Ex"]
    ax2.plot(
        xyz_finest[0],
        ex_interpolator(xyz_finest[0]),
        linewidth=2,
        color="dimgray",
    )
    ax2.set_ylim((-0.4, 0.4))
    ax1.set_ylim(-1, 2)

    ax2.set_ylabel("Ex(x)")
    ax1.set_ylabel("Vx - Velocity")
    ax1.set_xlabel("X - Position")

def phase_space_ey(time, r, **kwargs):
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)

    vmin, vmax = kwargs.get("vmin", -6), kwargs.get("vmin", 6)

    ions = r.GetParticles(time, ["main", "beam"])

    ions.dist_plot(
        axis=("x", "Vx"),
        ax=ax1,
        norm=0.4,
        finest=True,
        gaussian_filter_sigma=(1, 1),
        vmin=vmin,
        vmax=vmax,
        dv=0.05,
        title="t = {:.1f}".format(time),
        xlabel="",
        ylabel="",
    )

    ax2 = ax1.twinx()

    E_hier = r.GetE(time, merged=True, interp="linear")
    ey_interpolator, xyz_finest = E_hier["Ey"]
    ax2.plot(
        xyz_finest[0],
        ey_interpolator(xyz_finest[0]),
        linewidth=2,
        color="dimgray",
    )
    ax2.set_ylim((-0.4, 0.4))

    ax2.set_ylabel("Ey(x)")
    ax1.set_ylabel("Vx - Velocity")
    ax1.set_xlabel("X - Position")

def growth_plot(r,path, time_offset, **kwargs):
    times, first_mode, ampl, gamma, damped_mode, omega = growth_b_right_hand(path, time_offset, 0)
    file = os.path.join(r.path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)  # get_times_from_h5
    dt = times[1] - times[0]
    imax = find_peaks(first_mode, width=int(10 / dt))[0][
        0
    ]  # the smallest width of the peak is 10

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    ax1.set_title("Time evolution of the first right-hand circular mode amplitude")
    ax1.plot(
        times,
        first_mode,
        color="k",
        label="|$\hat{B}_y (m=1,t)-i\hat{B}_z (m=1,t)$|",
    )
    ax1.plot(
        times[:imax],
        yaebx(times[:imax], ampl, gamma),
        color="r",
        linestyle="-",
        label="$B_0. \exp(\gamma t), \ with\ \gamma =${:5.5f} (expected 0.09)".format(
            gamma
        ),
    )
    ax1.axvline(0, 0, yaebx(times[imax], ampl, gamma), color="red", linestyle="--")
    ax1.axvline(
        46,
        0,
        yaebx(times[imax], ampl, gamma),
        color="red",
        linestyle="--",
    )
    ax1.legend()
    ax1.set_xlabel("t - Time")

def xyz_fine(time, r, **kwargs):
    E_hier = r.GetE(time, merged=True, interp="nearest")
    ey_interpolator, xyz_finest = E_hier["Ey"]
    return xyz_finest

def modes_right(run_path, i):
    file = os.path.join(run_path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)  # Was get_times_from_h5 #Was none
    dt = times[1] - times[0]
    r = Run(run_path)
    modes = np.array([])
    for time in times:
        B_hier = r.GetB(time, merged=True, interp="nearest") #Was linear

        by_interpolator, xyz_finest = B_hier["By"]
        bz_interpolator, xyz_finest = B_hier["Bz"]

        # remove the last point so that "x" is periodic wo. last point = first point
        x = xyz_finest[0][:-1]

        by = by_interpolator(x)
        bz = bz_interpolator(x)

        # get the mode 1, as it is the most unstable in a box of length 33
        mode = np.absolute(np.fft.fft(by - 1j * bz)[i+1])
        modes = np.append(modes,mode)
    return modes, times

def modes_right_ib3(run_path, i):
    file = os.path.join(run_path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)  # Was get_times_from_h5 #Was none
    dt = times[1] - times[0]
    r = Run(run_path)
    modes = np.array([])
    for time in times:
        B_hier = r.GetB(time, merged=True, interp="nearest") #Was linear

        by_interpolator, xyz_finest = B_hier["By"]
        bz_interpolator, xyz_finest = B_hier["Bz"]

        # remove the last point so that "x" is periodic wo. last point = first point
        x = xyz_finest[0][:-1]

        by = by_interpolator(x)
        bz = bz_interpolator(x)

        # get the mode 1, as it is the most unstable in a box of length 33
        mode = np.absolute(np.fft.fft(by - 1j * bz)[i+1])
        modes = np.append(gf(modes, 1),mode)
    return modes, times
    
def modes_left(run_path, i):
    file = os.path.join(run_path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)  # Was get_times_from_h5 #Was none
    dt = times[1] - times[0]
    r = Run(run_path)
    modes = np.array([])
    for time in times:
        B_hier = r.GetB(time, merged=True, interp="nearest") #Was linear

        by_interpolator, xyz_finest = B_hier["By"]
        bz_interpolator, xyz_finest = B_hier["Bz"]

        # remove the last point so that "x" is periodic wo. last point = first point
        x = xyz_finest[0][:-1]

        by = by_interpolator(x)
        bz = bz_interpolator(x)

        # get the mode 1, as it is the most unstable in a box of length 33
        mode = np.absolute(np.fft.fft(by + 1j * bz)[i+1])
        modes = np.append(modes,mode)
    return modes, times


    
def modes_E(run_path, i):
    file = os.path.join(run_path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)  # Was get_times_from_h5 #Was none
    dt = times[1] - times[0]
    r = Run(run_path)
    modes = np.array([])
    for time in times:
        E_hier = r.GetE(time, merged=True, interp="nearest") #Was linear

        ex_interpolator, xyz_finest = E_hier["Ex"]

        # remove the last point so that "x" is periodic wo. last point = first point
        x = xyz_finest[0][:-1]

        ex = ex_interpolator(x)

        # get the mode 1, as it is the most unstable in a box of length 33
        mode = np.absolute(np.fft.fft(ex)[i+1])
        modes = np.append(gf(modes, 1),mode)
    return modes, times

def profiles(times, x, Vs, vmin=None, vmax=None, marker=None, **kw):
    fig, ax = plt.subplots()
    for it, t in enumerate(times):
        ax.plot(x, Vs[:, it], label=r"t={:5.2f}".format(t), marker=marker)
        ax.set_ylim((-1.5, 1.5))
        ax.set_xlim((5, 15))
        ax.axhline(0, ls="--", color="k")
        if vmin is not None and vmax is not None:
            ax.set_ylim((vmin, vmax))
    ax.legend(ncol=4)

def gamma_B(run_path, i):
    file = os.path.join(run_path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)
    dt = times[1] - times[0]
    r = Run(run_path)
    first_mode = np.array([])

    for time in times:
        B_hier = r.GetB(time, merged=True, interp="linear")

        by_interpolator, xyz_finest = B_hier["By"]
        bz_interpolator, xyz_finest = B_hier["Bz"]
        
        
        x = xyz_finest[0][:-1]

        by = by_interpolator(x)
        bz = bz_interpolator(x)

        mode1 = np.absolute(np.fft.fft(by - 1j * bz)[i+1])
        first_mode = np.append(first_mode, mode1)

    tmax = np.argmax(first_mode)
    xmax = first_mode[tmax]

    first_mode = first_mode[:tmax]
    times = times[:tmax]

    popt, pcov = curve_fit(yaebx, times[: tmax], first_mode, p0=[0.08, 0.09])
    a, b = popt
    
    return a, b, times

def gamma_Ex(run_path, i):
    file = os.path.join(run_path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)
    dt = times[1] - times[0]
    r = Run(run_path)
    first_mode = np.array([])

    for time in times:
        E_hier = r.GetE(time, merged=True, interp="nearest") #Was linear

        ex_interpolator, xyz_finest = E_hier["Ex"]

        # remove the last point so that "x" is periodic wo. last point = first point
        x = xyz_finest[0][:-1]

        ex = ex_interpolator(x)

        # get the mode 1, as it is the most unstable in a box of length 33
        mode = np.absolute(np.fft.fft(ex)[i+1])
        first_mode = np.append(modes,mode)
    print
    tmax = np.argmax(first_mode)
    xmax = first_mode[tmax]

    first_mode = first_mode[:tmax]
    times = times[:tmax]

    popt, pcov = curve_fit(yaebx, times[: tmax], first_mode, p0=[0.08, 0.09])
    a, b = popt
    
    return a, b, times

def growth_ib3(r, run_path,**kwargs):
    "only for ib3.py"
    time_offset = 23
    file = os.path.join(run_path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)
    dt = times[1] - times[0]
    r = Run(run_path)
    first_mode = np.array([])

    for time in times:
        B_hier = r.GetB(time, merged=True, interp="linear")

        by_interpolator, xyz_finest = B_hier["By"]
        bz_interpolator, xyz_finest = B_hier["Bz"]
        
        x = xyz_finest[0][:-1]

        by = by_interpolator(x)
        bz = bz_interpolator(x)

        mode1 = np.absolute(np.fft.fft(by - 1j * bz)[1])
        first_mode = np.append(gf(first_mode, 3), mode1)


    ioffset = int(time_offset / dt)
    imax = find_peaks(first_mode, width=ioffset)[0][0]

    popt, pcov = curve_fit(yaebx, times[: imax - ioffset], first_mode[: imax - ioffset], p0=[0.08, 0.09])
    ampl, gamma = popt

    damped_mode = first_mode[: imax - ioffset] * yaebx(times[: imax - ioffset], 1 / popt[0], -popt[1])

    omegas = np.fabs(np.fft.fft(damped_mode).real)
    omega = (0.5 * (omegas[1 : omegas.size // 2].argmax() + 1) * 2 * np.pi / times[imax - 1 - ioffset])
    
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    ax1.set_title("Time evolution of the first right-hand circular mode amplitude")
    ax1.plot(
        times,
        first_mode,
        color="k",
        label="|$\hat{B}_y$|",
    )
    ax1.plot(
        times[:imax],
        yaebx(times[:imax], ampl, gamma),
        color="r",
        linestyle="-",
        label="$B_0. \exp(\gamma t), \ with\ \gamma =${:5.5f} (expected 0.09)".format(
            gamma
        ),
    )
    ax1.axvline(0, 0, yaebx(times[imax], ampl, gamma), color="red", linestyle="--")
    ax1.axvline(
        times[imax] - time_offset,
        0,
        yaebx(times[imax], ampl, gamma),
        color="red",
        linestyle="--",
    )
    ax1.legend()
    ax1.set_xlabel("t - Time")

    return times, first_mode, ampl, gamma, damped_mode, omega

def growth_ion_acoustic(r, run_path,**kwargs):
    "only for ib1.py"
    time_offset = 20
    file = os.path.join(run_path, "EM_B.h5")
    times = fromh5.get_times_from_h5(file)
    dt = times[1] - times[0]
    r = Run(run_path)
    first_mode = np.array([])

    for time in times:
        E_hier = r.GetE(time, merged=True, interp="linear")

        ex_interpolator, xyz_finest = E_hier["Ex"]
        
        x = xyz_finest[0][:-1]

        ex = ex_interpolator(x)

        mode1 = np.absolute(np.fft.fft(ex)[6])
        first_mode = np.append(gf(first_mode, 1), mode1)

    ioffset = int(time_offset / dt)
    imax = find_peaks(first_mode, width=ioffset)[0][0]

    popt, pcov = curve_fit(yaebx, times[: imax - ioffset], first_mode[: imax - ioffset], p0=[0.1, 0.2])
    ampl, gamma = popt

    damped_mode = first_mode[: imax - ioffset] * yaebx(times[: imax - ioffset], 1 / popt[0], -popt[1])

    omegas = np.fabs(np.fft.fft(damped_mode).real)
    omega = (0.5 * (omegas[1 : omegas.size // 2].argmax() + 1) * 2 * np.pi / times[imax - 1 - ioffset])
    
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)

    ax1.set_title("Time evolution of the ion acoustic mode amplitude")
    ax1.plot(
        times,
        first_mode,
        color="k",
        label="|$\hat{E}_x$|",
    )
    ax1.plot(
        times[:imax],
        yaebx(times[:imax], ampl, gamma),
        color="r",
        linestyle="-",
        label="$E_0. \exp(\gamma t), \ with\ \gamma =${:5.5f}".format(
            gamma
        ),
    )
    ax1.legend()
    ax1.set_xlabel("t - Time")

    ax1.set_ylim(0, 0.7)
    
    return times, first_mode, ampl, gamma, damped_mode, omega