import os

from pyphare.simulator.simulator import Simulator
from pyphare.pharesee.run import Run

import pyphare.pharein as ph

from pyphare.pharein import global_vars
from tests.diagnostic import all_timestamps

from pyphare.pharein.diagnostics import FluidDiagnostics

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.use("Agg")


# L = sim.simulation_domain()[0]
L = 20.0

m_Main = 2
c_Main = 1
m_Beam = 3
c_Beam = 2


def densityMain(x):
    return 1.0

def densityBeam(x):
    u = x/L-0.5
    return np.exp(-u**2)

def bx(x):
    return 1.0

def by(x):
    return 0.0

def bz(x):
    return 0.0

def v0(x):
    return 0.0

def vth(x):
    return np.sqrt(1.0)


def config():

    sim = ph.Simulation(
        smallest_patch_size=20,
        largest_patch_size=60,
        time_step=0.01,
        time_step_nbr=1,
        boundary_types="periodic",
        cells=100,
        dl=0.2,
        diag_options={
            "format": "phareh5",
            "options": {"dir": "nCheck", "mode": "overwrite"},
        },
    )

    v_pop = {
        "vbulkx": v0,
        "vbulky": v0,
        "vbulkz": v0,
        "vthx": vth,
        "vthy": vth,
        "vthz": vth,
    }

    ph.MaxwellianFluidModel(
        bx=bx,
        by=by,
        bz=bz,
        main={"mass": m_Main, "charge": c_Main, "density": densityMain, "nbr_part_per_cell": 1000, **v_pop},
        beam={"mass": m_Beam, "charge": c_Beam, "density": densityBeam, "nbr_part_per_cell": 1000, **v_pop},
    )

    ph.ElectronModel(closure="isothermal", Te=0.0)

    timestamps = all_timestamps(global_vars.sim)

    for quantity in ["charge_density", "mass_density"]:
        FluidDiagnostics(
            quantity=quantity,
            write_timestamps=timestamps
        )

    poplist = ["main", "beam"]
    for pop in poplist:
        for quantity in ["density", "charge_density"]:
            FluidDiagnostics(
                quantity=quantity,
                write_timestamps=timestamps,
                population_name=pop,
            )

    return sim


# def _compute_ions_mass_density_profile(patch_datas, **kwargs):
#     ref_name = next(iter(patch_datas.keys()))
# 
#     x_ = patch_datas[ref_name].x
#     # print(type(x_), x_)
#     dset = m_Main*densityMain(x_)+m_Beam*densityBeam(x_)
# 
#     # dset = patch_datas["value"][:]
#     return ({"name": "value", "data": dset, "centering": patch_datas[ref_name].centerings},)


def main():
    Simulator(config()).run()

    run_path = os.path.join(os.curdir, "nCheck")
    time = 0.0
    r = Run(run_path)
    mass_computed = r.GetMassDensity(time)
    charge_computed = r.GetNi(time)

    # from pyphare.pharesee.hierarchy.hierarchy_utils import compute_hier_from
    # mass_expected = compute_hier_from(_compute_ions_mass_density_profile, mass_computed)


    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    titles = ["Mass density", "Charge density"]

    for lvl_m, lvl_c in zip(mass_computed.levels(time).values(), charge_computed.levels(time).values()):
        for patch_m, patch_c in zip(lvl_m.patches, lvl_c.patches):
            pd_m = patch_m.patch_datas["value"]
            pd_c = patch_c.patch_datas["value"]
            ghosts_num = pd_m.ghosts_nbr[0]

            computed_mass = pd_m.dataset[ghosts_num:-ghosts_num]
            computed_charge= pd_c.dataset[ghosts_num:-ghosts_num]
            x_ = pd_m.x[ghosts_num:-ghosts_num]
            expected_mass = m_Main*densityMain(x_)+m_Beam*densityBeam(x_)
            expected_charge = c_Main*densityMain(x_)+c_Beam*densityBeam(x_)

            for c, e in zip(computed_mass, expected_mass):
                np.testing.assert_almost_equal(c, e, decimal=1)

            for c, e in zip(computed_charge, expected_charge):
                np.testing.assert_almost_equal(c, e, decimal=1)

            ax[0].plot(x_, computed_mass, color=cycle[0])
            ax[0].plot(x_, expected_mass, color=cycle[1])
            ax[1].plot(x_, computed_charge, color=cycle[0])
            ax[1].plot(x_, expected_charge, color=cycle[1])

            for i in range(2):
                ax[i].legend(('Computed', 'Expected'), loc='lower center', shadow=True)
                ax[i].set_xlabel("x")
                ax[i].set_ylabel("rho")
                ax[i].set_xlim([0, 20])
                # ax[i].set_ylim([4, 5.4])
                ax[i].set_title(titles[i])

    plt.savefig("test_density.pdf", dpi=300)






    # for lvl in mass_computed.levels(time).values():
    #     for patch in lvl.patches:
    #         pd = patch.patch_datas["value"]
    #         ghosts_num = pd.ghosts_nbr[0]

    #         computed = pd.dataset[ghosts_num:-ghosts_num]
    #         x_ = pd.x[ghosts_num:-ghosts_num]
    #         expected = m_Main*densityMain(x_)+m_Beam*densityBeam(x_)

    #         for c, e in zip(computed, expected):
    #             np.testing.assert_almost_equal(c, e, decimal=1)

    #         ax[0].plot(x_, computed, color=cycle[0])
    #         ax[0].plot(x_, expected, color=cycle[1])
    #         ax[0].legend(('Computed', 'Expected'), loc='lower center', shadow=True)
    #         ax[0].set_xlabel("x")
    #         ax[0].set_ylabel("rho")
    #         ax[0].set_xlim([0, 20])
    #         ax[0].set_ylim([4, 5.4])
    #         ax[0].set_title("charge density")

    # plt.savefig("test_density.pdf", dpi=300)

if __name__ == "__main__":
    main()

