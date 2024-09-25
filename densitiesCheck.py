import os

import pyphare.pharein as ph
from pyphare.simulator.simulator import Simulator
from pyphare.pharesee.hierarchy.fromh5 import get_times_from_h5
from pyphare.pharesee.run import Run

from pyphare.pharein.diagnostics import FluidDiagnostics

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


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
            "options": {"dir": "nCheck", "mode": "overwrite"},
        },
    )

    def densityMain(x):
        return 1.0

    def densityBeam(x):
        return 0.1

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
        main={"mass": 1, "charge": 1, "density": densityMain, **v_pop},
        beam={"mass": 1, "charge": 1, "density": densityBeam, **v_pop},
    )

    ph.ElectronModel(closure="isothermal", Te=0.0)

    for quantity in ["density", "charge_density"]:
        FluidDiagnostics(
            quantity=quantity, write_timestamps=np.zeros(time_step_nbr)
        )

    poplist = ["main", "beam"]
    for pop in poplist:
        for quantity in ["density", "charge_density", "mass_density"]:
            FluidDiagnostics(
                quantity=quantity,
                write_timestamps=np.zeros(time_step_nbr),
                population_name=pop,
            )

    return sim


def get_hierarchies(run_path):
    r = Run(run_path)

    N_hier = r.GetNi(0.0)

    return N_hier


def main():
    from pybindlibs.cpp import mpi_rank

    Simulator(config()).run()

    if mpi_rank() == 0:
        n_hier = get_hierarchies( os.path.join(os.curdir, "nCheck") )

        assert np.fabs(0.01) < 0.1


if __name__ == "__main__":
    main()
