import matplotlib.pyplot as plt

import soundscapy as sspy

coords = sspy.project(sspy.load_isd(subset=True), angles="equal")
ax = sspy.plot.scatter(coords, by="location")
plt.savefig("first-plot.png")
