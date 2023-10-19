import pyabf
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


@dataclass
class ExtractedAbf:
    area: float
    x: list[float]
    y_avg: np.ndarray[float]


def window(size):
    return np.ones(size) / float(size)


def find_roots(x, y):
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)


def exponential_f(x, a, b, c):
    return a * np.exp(-b * x) + c


def generate_mean(x, ys: list[float], conv_wid=500, plot=True, axis=plt.gca()):
    y = np.mean(ys, axis=0)

    y_avg = np.convolve(y, window(conv_wid), 'same')
    if plot:
        axis.plot(x, y, color='r', alpha=0.3, label='Mean')
        line, = axis.plot(x, y_avg, color='k', label=f'Convolved Mean (n={conv_wid})')
    else:
        line = None
    return y_avg, line


def get_root(x, y, start_range, plot=True, axis=plt.gca()):
    # find first crossing of 0 
    roots = find_roots(x, y)
    root = roots[np.ma.masked_outside(roots, start_range[0], start_range[1]).mask is False]
    if plot:
        axis.axvline(x=root, color='k', alpha=0.3, linestyle='--')

    return root


def fit_integrate(x, y, intg_start=0.5, rate=20000, plot=True, axis=plt.gca()):
    xy = list(zip(x, y))

    # find minimum
    mini = min(xy, key=lambda p: p[1])
    mask = x > mini[0]

    # integrate
    mask = x > intg_start
    x_range = x[mask]
    to_integrate = np.minimum(y[mask], np.zeros(y[mask].shape))  # y[mask]
    area = np.trapz(to_integrate, dx=1 / rate)

    if plot:
        axis.fill_between(x_range, to_integrate, alpha=0.3, label=f"Area: {area:.2f}mVs")

    return area


def gen_customticks(rng, bel, abv):
    return [-bel * i for i in range(round(rng[0] / -bel) + 1)] + [abv * i for i in range(1, round(rng[1] / abv) + 1)]


START_RANGE = [0.06, 0.2]


def setup_graph(ys, grph, custom_scale):
    grph.set_xlabel("Time (s)")
    grph.set_ylabel("Voltage (mV)")
    grph.set_xlim(0, 2)
    grph.legend(loc='upper right')
    grph.grid(which='major')
    grph.axhline(linewidth=1, color='gray')  # thickened y=0 line
    grph.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)

    rng = (int(np.min(ys)), int(np.max(ys)))

    if custom_scale:
        scaling_above = 20
        # these functions scale values above 0 by scaling_above (kinda gross)
        grph.set_yscale("function", functions=[lambda x: np.where(x > 0, x / scaling_above, x),
                                               lambda x: np.where(x > 0, x * scaling_above, x)])

        maj_yticks = gen_customticks(rng, 5,
                                     5 * scaling_above)  # ticks every 5 beneath 0, every 5*scaling above above 0
        min_yticks = gen_customticks(rng, 1, scaling_above)
        min_xticks = gen_customticks((0, 2), 1, 0.125)  # can't rely on the autogenerator because of the weird scaling

        grph.set_yticks(maj_yticks, major=True)
        grph.set_yticks(min_yticks, minor=True)
        grph.set_xticks(min_xticks, minor=True)
    else:
        grph.minorticks_on()
        grph.set_ylim(min(((rng[0]) // 5) * 5, -5), 10)


def process(abf: pyabf.ABF, grph, from_root=False, integration_start=0.5, conv_wid=100, custom_scale=False) -> ExtractedAbf:
    x = abf.sweepX
    ys = plot_abf(abf, grph, x)

    return extract_and_plot(x, ys, grph, conv_wid, abf.sampleRate, integration_start, from_root, custom_scale)


def plot_abf(abf: pyabf.ABF, grph, x: np.ndarray[float], baseline_bounds=[0.01, 0.03]):
    ys = []
    for sweep in range(len(abf.sweepTimesSec)):
        if baseline_bounds is not None:
            abf.setSweep(sweep, baseline=baseline_bounds)
        else:
            abf.setSweep(sweep)

        if len(abf.sweepY) == len(x):
            ys.append(abf.sweepY)
        else:
            break

        grph.plot(abf.sweepX, abf.sweepY, alpha=.2, color=f"C{sweep}")

    return ys


def process_multiple(abf_names, grph, from_root=False, integration_start=0.5, conv_wid=100, custom_scale=False) -> ExtractedAbf:
    # get and plot all sweeps
    ys = []
    x = []
    sample_rate = 10

    for abf_file in abf_names:
        abf = pyabf.ABF(abf_file)
        sample_rate = abf.sampleRate

        if len(abf.sweepX) == 40000:
            x = abf.sweepX

        ys += plot_abf(abf, grph, x)

    return extract_and_plot(x, ys, grph, conv_wid, sample_rate, integration_start, from_root, custom_scale)


def extract_and_plot(x, ys, grph, conv_wid, sample_rate, integration_start, from_root, custom_scale):
    y_avg, line = generate_mean(x, ys, conv_wid, plot=True, axis=grph)

    if from_root:
        roots = find_roots(x, y_avg)
        found_root = next((root for root in roots if (START_RANGE[0] < root < START_RANGE[1])), None)

        if found_root is not None:
            integration_start = found_root
        else:
            # area = 0
            setup_graph(ys, grph, custom_scale)
            return ExtractedAbf(0, x, y_avg)

    area = fit_integrate(x, y_avg, intg_start=integration_start, rate=sample_rate, plot=True, axis=grph)
    setup_graph(ys, grph, custom_scale)
    return ExtractedAbf(area, x, y_avg)
