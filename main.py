from utils import process, process_multiple, exponential_f
import pandas as pd
import pyabf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.widgets import CheckButtons, Button

class Manager:
    def __init__(self, files, grph, cx, fig1):
        self.files = files
        self.grph = grph 
        self.cx = cx
        self.fig1 = fig1
        self.idx = 0
        self.excluded = set() 
        self.baseline = True
        self.baseline_bounds = [0.01, 0.03]

    def move(self, event):
        self.save_current()
        self.skip(None)

    def skip(self, event):
        self.idx += 1
        if self.idx >= len(files):
            exit()
        self.showfile()

    def save_current(self):
        '''
            construct and pandas df to contain the averaged data, integral, and fitted parameters?
            save figure containing graph
        '''
        self.grph.clear()
        sweeps, ((x,y), coeff, area) = process(self.abf, self.grph,
                excluded=self.excluded,
                baseline_bounds=self.baseline_bounds if self.baseline else None)

        data = {'Time (s)': x,
                'Voltage (mV)': y,
                'Area past 500ms (mVs)': [area] + [""]*(len(x)-1)}
        df = pd.DataFrame(data) 
        file = self.files[self.idx]
        df.to_csv(f'{file.parent/file.stem}.csv', index=False)

        self.fig1.savefig(f'{file.parent/file.stem}.png')
        print(file)

    def showfile(self):
        self.abf = pyabf.ABF(self.files[self.idx])
        self.grph.clear()
        sweeps, _ = process(self.abf, self.grph,
                baseline_bounds=self.baseline_bounds if self.baseline else None)

        self.cx.clear()
        self.generate_checkboxes(sweeps)
        self.fig1.suptitle(self.files[self.idx])
        self.fig1.canvas.draw_idle()
        #self.move(None)

    def generate_checkboxes(self, sweeps):
        num_sweeps = len(sweeps)

        self.labels = [f"Sweep {i+1}" for i in range(num_sweeps)]
        self.check = CheckButtons(self.cx, self.labels, [True]*num_sweeps)

        self.check.on_clicked(self.update_sweeps)
        # set colors to match the lines they toggle
        [rec.set_facecolor(sweeps[i].get_color()) for i, rec in enumerate(self.check.rectangles)]

    def update_sweeps(self, label):
        idx = self.labels.index(label)
        if idx in self.excluded:
            self.excluded.remove(idx)
        elif len(self.excluded) < len(self.labels)-1:
            self.excluded.add(idx)
        else:
            self.fig1.canvas.draw_idle()
            return
        self.update()

    def baselinetoggle(self, label):
        self.baseline = not self.baseline
        self.update()

    def update(self):
        self.grph.clear()
        process(self.abf, self.grph,
                excluded=self.excluded, 
                baseline_bounds=self.baseline_bounds if self.baseline else None)
        self.fig1.canvas.draw_idle()

if __name__ == "__main__":

    #filepath = "data/21729016.abf"
    p = Path('./data/').glob('**/AHP/*.abf')
    files = [x for x in p if x.is_file()]
#    files = [Path("/home/linus/Documents/abf_avg/data/210729 Shane 183562 8 mo old female Tg/cell4/AHP/21729016.abf")]
#    wt_bvk = [f for f in files if "WT" in f and "VK" not in f]
#    tg_bvk = [f for f in files if "Tg" in f and "VK" not in f]
#    wt_avk = [f for f in files if "WT" in f and "VK" in f]
#    tg_avk = [f for f in files if "Tg" in f and "VK" in f]

#    groups = [wt_bvk, tg_bvk, wt_avk, tg_avk]
#    labels = ["WT", "Tg", "WT VK", "Tg VK"]

    abf = pyabf.ABF(files[0])
    plt.close() # close figure created by pyABF!
    fig, grph = plt.subplots(1)
    fig2, cx = plt.subplots(1)
#
#    fig.canvas.manager.set_window_title("Graph")
#    y_avgs = []
#    xs = []
#    areas = []
#    params = []
#    pcovs = []
#    for group in groups:
#        sweeps, ((x, y_avg), (param, pcov), area) = process_multiple(group, grph)
#        y_avgs.append(y_avg)
#        xs.append(x)
#        areas.append(area)
#        params.append(param)
#        pcovs.append(pcov)
#
#
#    plots = {
#        "WT_BF_AF": {"idxs": (0, 2)},
#        "TG_BF_AF": {"idxs": (1, 3)},
#        "WT_TG_BF": {"idxs": (0, 1)}
#    }
#    grph.clear()
#    for name in plots:
#        indices = plots[name]['idxs']
#        with open(f"{name}.csv", 'w') as f:
#            f.write("\n".join(f"{x},{y1},{y2}" for x, y1, y2 in zip(xs[indices[0]], y_avgs[indices[0]], y_avgs[indices[1]])))
#
#        for idx1, idx in enumerate(indices):
#
#            grph.plot(xs[idx], y_avgs[idx], label=f"{labels[idx]} (A={areas[idx]:.2f}mVs, n={len(groups[idx])})")
#    
#            xy = list(zip(xs[idx], y_avgs[idx]))
#
#            mini = min(xy, key=lambda p: p[1])
#            mask = x > mini[0]
#
#            exp = exponential_f(x[mask], *params[idx])
#            chi_squared = np.sum(np.power(y_avgs[idx][mask]-exp, 2)/exp)
#    
#    
#            grph.plot(x[mask], exponential_f(x[mask], *params[idx]),
#                      linestyle='--',
#                      color=f'C{idx1}',
#                      alpha=0.6,
#                      label=labels[idx] + " ({:.2f}e$^{{-{:.2f}x}}$ {:+.2f})), $\chi^2$={:.2f}".format(*params[idx], chi_squared))
#
#        grph.set_xlabel("Time (s)")
#        grph.set_ylabel("Voltage (mV)")
#        grph.set_xlim(0, 2)
#        grph.legend(loc='best')
#        grph.axhline(linewidth=1, color='gray') # thickened y=0 line
#        #grph.grid(which='major')
#        #grph.grid(which='minor', color='gray', linestyle='-', linewidth=0.1)
#
#        rng = (int(np.min([y_avgs[indices[0]], y_avgs[indices[1]]])), int(np.max([y_avgs[indices[0]], y_avgs[indices[1]]])))
#        grph.minorticks_on()
#        grph.set_ylim(rng[0]-1, 2)
#        fig.suptitle(f"{labels[indices[0]]} vs {labels[indices[1]]}")
#
#        fig.savefig(f"plot_{name}.png")
#        grph.clear()
#
    

    fig2.canvas.manager.set_window_title("Controls")
    
    manager = Manager(files, grph, cx, fig)
    manager.showfile()

    # save button/triggering everything because?
    buttax = fig2.add_axes([0.7, 0.15, 0.15, 0.1])
    save = Button(buttax, "Save")
    save.on_clicked(manager.move)

    # skip button
    skipax = fig2.add_axes([0.5, 0.15, 0.15, 0.1])
    skip = Button(skipax, "Skip")
    skip.on_clicked(manager.skip)

    # baseline correction
    baseax = fig2.add_axes([0.6, 0.7, 0.3, 0.2])
    basecheck = CheckButtons(baseax, ['Baseline Correction'], [True])

    basecheck.on_clicked(manager.baselinetoggle)

    plt.show()
