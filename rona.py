import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from jupyterthemes import jtplot

jtplot.style(
    theme="onedork",
    context="talk",
    fscale=1.5,
    spines=True,
    gridlines="--",
    ticks=True,
    grid=False,
)


class Covid:
    def __init__(self, y0, params):
        # initial condition
        self.y0 = y0
        self.N = np.sum(y0)  # total population

        # transmission dynamics
        self.R0 = params[0]  # reproduction number
        self.Rt = params[1]  # transmission number
        self.Tinc = params[2]  # incubation period [days]
        self.Tinf = params[3]  # infectious period  [days]
        self.Tlock = params[4]  # lockdown [days]

        # clinical dynamics
        self.mortality = 0.02  # case fatality rate
        self.death_time = 32  # time from end of incubation to death [days]
        self.recovery_time_hospital = 28  # length of hospital stay [days]
        self.recovery_time_mild = 11  # recovery time for mild cases [days]
        self.hospitalization_rate = 0.2  # hospitalization rate
        self.hospitalization_time = 5  # time to hospitalization  [days]

    def seir(self, t, y):
        # susceptible, exposed, infected, recovered
        S, E, I, R = y

        if t < self.Tlock:
            Rt = self.R0
        else:
            Rt = self.Rt

        dSdt = -(S / self.N) * (Rt * I / self.Tinf)
        dEdt = (S / self.N) * (Rt * I / self.Tinf) - E / self.Tinc
        dIdt = E / self.Tinc - I / self.Tinf
        dRdt = I / self.Tinf

        return np.array([dSdt, dEdt, dIdt, dRdt])

    def solve(self, tmin, tmax):
        self.tmin = tmin
        self.tmax = tmax
        self.sol = solve_ivp(
            self.seir,
            t_span=(self.tmin, self.tmax),
            y0=self.y0,
            vectorized=True,
            dense_output=True,
        )

    def plot(self):
        t = np.arange(self.tmin, self.tmax + 1, 1)
        self.z = self.sol.sol(t)
        infected = self.z[2, :]
        self.z_cum = np.cumsum(infected)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(t, self.z.T[:, [1, 2]])
        ax.axhline(y=800, color='gray', ls='--', label="__nolegend__")
        ax.axvline(x=self.Tlock, color="gray", ls=":", label="__nolegend__")
        ax.set_xlim(left=self.tmin, right=self.tmax)
        ax.legend(["Exposed", "Infected"], loc="upper left")
        ax.set_xlabel("days")
        ax.ticklabel_format(useMathText=True, scilimits=(0, 0), axis="y")
        ax.spines["top"].set_visible(False)

        twax = plt.twinx(ax)
        twax.plot(t, self.z_cum / self.N, color="orange", ls="--")
        twax.spines["top"].set_visible(False)
        # twax.ticklabel_format(useMathText=True, scilimits=(0, 0), axis="y")
        plt.tight_layout()
        plt.show()

    def plot_hist(self):
        x = np.arange(self.tmin, self.tmax + 1, 1)
        weights = cv.z[2, :]
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.axhline(y=800, color='gray', ls='--', label="__nolegend__")
        ax.axvline(x=self.Tlock, color="gray", ls=":", label="__nolegend__")
        twax = ax.twinx()
        sns.distplot(
            x,
            bins=len(x) // 2,
            kde=False,
            ax=ax,
            hist_kws={
                "weights": weights,
                "alpha": 1.0,
                "rwidth": 0.8,
                "cumulative": False,
            },
        )
        sns.distplot(
            x,
            bins=len(x) // 2,
            kde=False,
            ax=twax,
            hist_kws={
                "weights": weights / np.sum(y0),
                "alpha": 0.5,
                "rwidth": 0.8,
                "cumulative": True,
                "color": "orange",
            },
        )
        sns.despine(right=False)
        ax.set_xlabel("days")
        ax.set_xlim(left=0, right=300)
        ax.ticklabel_format(useMathText=True, scilimits=(0, 0), axis="y")
        # twax.ticklabel_format(useMathText=True, scilimits=(0, 0), axis="y")
        plt.tight_layout()
        plt.show()


# y0 = [S, E, I, R]
y0 = np.array([1e7, 0, 1, 0])
# params = [R0, Rt, Tinc, Tinf, Tlock]
params = np.array([2.10, 0.75, 5.20, 2.90, 100])

tmin = 0
tmax = 365

cv = Covid(y0, params)
cv.solve(tmin, tmax)
cv.plot()
cv.plot_hist()
