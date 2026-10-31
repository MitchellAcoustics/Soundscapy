import soundscapy as sspy

coords = sspy.project(sspy.load_isd(subset=True), angles="equal")
sspy.plot.scatter(coords, by="location").figure.savefig("first-plot.png")
