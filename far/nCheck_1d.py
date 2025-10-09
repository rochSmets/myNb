import os

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator
from pyphare.pharesee.run import Run

from pyphare.pharein import global_vars
from tests.diagnostic import all_timestamps

from pyphare.pharein.diagnostics import FluidDiagnostics

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.use("Agg")


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
            "options": {"dir": "nCheck_1d", "mode": "overwrite"},
        },
    )

    L = sim.simulation_domain()[0]

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
        main={"mass": 2, "charge": 1, "density": densityMain, "nbr_part_per_cell": 1000, **v_pop},
        beam={"mass": 3, "charge": 2, "density": densityBeam, "nbr_part_per_cell": 1000, **v_pop},
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


def main():
    Simulator(config()).run()


if __name__ == "__main__":
    main()
