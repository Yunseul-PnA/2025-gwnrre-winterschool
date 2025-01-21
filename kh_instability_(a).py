"""
To run and plot using e.g. 4 processes:

    $ mpiexec -n 4 python3 kh_instability.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5

"""

import numpy as np
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)

###########################  Simulation parameters  ############################

Nx, Ny = 384, 384
stop_sim_time = 2.0

Reynolds = 1e4

################################################################################

Lx, Ly = 1, 1
Schmidt = 1
dealias = 3 / 2
timestepper = d3.SBDF3
max_timestep = 5e-3
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates("x", "y")
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.RealFourier(coords["y"], size=Ny, bounds=(-Ly / 2, Ly / 2), dealias=dealias)

# Fields
rho = dist.Field(name="rho", bases=(xbasis, ybasis))
p = dist.Field(name="p", bases=(xbasis, ybasis))
s = dist.Field(name="s", bases=(xbasis, ybasis))
u = dist.VectorField(coords, name="u", bases=(xbasis, ybasis))
tau_p = dist.Field(name="tau_p")

# Substitutions
nu = 1 / Reynolds
D = nu / Schmidt
x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)

# Problem
problem = d3.IVP([u, s, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u)")
problem.add_equation("dt(s) - D*lap(s) = - u@grad(s)")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial fluid velocity
delta_shear = 5e-4
u["g"][0] = 0.5 * (
    1.0 + np.tanh((y - 0.25) / delta_shear) + np.tanh(-(y + 0.25) / delta_shear)
)
u["g"][1] = 5e-3 * np.sin(2* np.pi * 2 * x) * (
    np.exp(-(y - 0.25)**2 / delta_shear) + np.exp(-(y + 0.25)**2 / delta_shear)
)

# Match tracer to shear
s["g"] = u["g"][0]

#######################################################################

# Add small vertical velocity perturbations localized to the shear layers
# u["g"][1] = ...


#######################################################################


# Saving simulation data
snapshots = solver.evaluator.add_file_handler("snapshots", sim_dt=0.1, max_writes=100)
snapshots.add_task(s, name="tracer")
snapshots.add_task(p, name="pressure")
snapshots.add_task(-d3.div(d3.skew(u)), name="vorticity")
snapshots.add_task(u @ ex, layout="g", name="vx")
snapshots.add_task(u @ ey, layout="g", name="vy")

analysis = solver.evaluator.add_file_handler("analysis", iter=20, max_writes=10000)
# integral of |vy|
analysis.add_task(
    d3.Integrate(np.abs(u @ ey), ("x", "y")), layout="g", name="abs_vy_integral"
)

# CFL
CFL = d3.CFL(
    solver,
    initial_dt=1e-5,
    cadence=10,
    safety=0.2,
    threshold=0.1,
    max_change=1.5,
    min_change=0.5,
    max_dt=max_timestep,
)
CFL.add_velocity(u)

# Main loop
try:
    logger.info("Starting main loop")
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration - 1) % 10 == 0:
            logger.info(
                "Iteration=%i, Time=%e, dt=%e"
                % (solver.iteration, solver.sim_time, timestep)
            )
except:
    logger.error("Exception raised, triggering end of main loop.")
    raise
finally:
    solver.log_stats()
