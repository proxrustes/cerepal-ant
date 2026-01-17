import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cube_track_YYYYMMDD_HHMMSS.csv")

plt.figure()
plt.plot(df["x"], df["z"])   # например top-down: X vs Z
plt.xlabel("x (m)")
plt.ylabel("z (m)")
plt.title("Trajectory (top-down)")
plt.axis("equal")
plt.show()
