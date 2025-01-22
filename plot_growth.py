import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

import h5py

file = h5py.File("analysis/analysis_s1.h5", mode="r")

abs_vy_integral = file["tasks"]["abs_vy_integral"]
t = abs_vy_integral.dims[0]["sim_time"][()]

plt.figure(figsize=(5, 3), dpi=130)

# plt.plot(t, abs_vy_integral[:, 0, 0])

#############################
# .... (add your own lines below to manipulate or add plots)

plt.plot(t, np.log(abs_vy_integral[:, 0, 0]), label = "Numerical Experiment")
print((np.log(abs_vy_integral[:, 0, 0])[60]-np.log(abs_vy_integral[:, 0, 0])[0])/(t[60]-t[0]))
# plt.plot([0,1],np.log(np.array([abs_vy_integral[0,0,0],abs_vy_integral[1,0,0]])))
# print((np.log(abs_vy_integral[:,0,0])[0]-np.log(abs_vy_integral[:,0,0])[60])/1)
plt.plot(t, 2 * np.pi * t, "--", label="Theoretical Value")
plt.legend()

#############################

plt.xlabel("Time", labelpad=10)
plt.ylabel(r"$lnI$", labelpad=10)

plt.show()

plt.savefig("InstabilityGrowth_ln.png", bbox_inches="tight", dpi=200)

file.close()
