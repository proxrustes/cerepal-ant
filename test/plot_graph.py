import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv(path: str):
    xs, ys, zs = [], [], []
    ts = []
    held = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ts.append(int(row["ts_ns"]))
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
            held.append(row.get("held", "false").lower() == "true")
    return np.array(ts), np.array(xs), np.array(ys), np.array(zs), np.array(held)

def set_axes_equal(ax):
    # чтобы 3D не был “растянутым”
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

def main():
    path = "./tracking/latest.csv"  # <-- твой файл
    ts, x, y, z, held = load_csv(path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # линия траектории
    ax.plot(x, y, z, linewidth=2)

    # старт/финиш
    ax.scatter([x[0]], [y[0]], [z[0]], s=50, marker="o")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], s=50, marker="x")

    # подписи осей
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectory (cube frame)")

    set_axes_equal(ax)
    plt.show()

if __name__ == "__main__":
    main()
