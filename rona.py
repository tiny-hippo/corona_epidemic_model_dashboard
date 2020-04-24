import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
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
        self.N = y0[0]
        self.I0 = y0[1]
        self.inital_state = np.array(
            [1 - self.I0 / self.N, 0, self.I0 / self.N, 0, 0, 0, 0, 0, 0, 0]
        )

        # transmission dynamics
        self.R0 = params[0]  # reproduction number
        self.OMInterventionAmt = params[1]  # transmission number
        self.InterventionAmt = 1 - self.OMInterventionAmt
        self.D_incubation = params[2]  # incubation period [days]
        self.D_infectious = params[3]  # infectious period  [days]
        self.D_lockdown = params[4]  # lockdown start [days]
        self.D_lockdown_duration = 7 * 12 * 1e10  # lockdown duration [days]

        # clinical dynamics
        self.time_to_death = 32
        self.D_recovery_mild = 11.1  # recovery time for mild cases [days]
        self.D_recovery_severe = 28.6  # length of hospital stay [days]
        self.D_hospital_lag = 5  # # time to hospitalization [days]
        self.D_death = (
            self.time_to_death - self.D_infectious
        )  #  time from end of incubation to death [days]
        self.cfr = 0.02  # case fatality rate
        self.p_severe = 0.2  # hospitalization rate

    def seir_expanded(self, t, y):
        # susceptible, exposed, infectious, recovering (mild),
        # recovering (severe at home), recovering (severe in hospital),
        # recovering (fatal), recovered,(mild), recovered (severe), dead
        S, E, I, Mild, Sev, Sev_H, F, R_Mild, R_Sev, R_F = y

        # assert abs(np.sum(y) - 1) < 1e-7

        if t > self.D_lockdown and t < self.D_lockdown + self.D_lockdown_duration:
            beta = self.InterventionAmt * self.R0 / self.D_infectious
        elif t > self.D_lockdown_duration + self.D_lockdown_duration:
            beta = 0.5 * self.R0 / self.D_infectious
        else:
            beta = self.R0 / self.D_infectious

        a = 1 / self.D_incubation
        gamma = 1 / self.D_infectious

        p_severe = self.p_severe
        p_fatal = self.cfr
        p_mild = 1 - p_severe - p_fatal

        dS = -beta * I * S
        dE = beta * I * S - a * E
        dI = a * E - gamma * I
        dMild = p_mild * gamma * I - (1 / self.D_recovery_mild) * Mild
        dSev = p_severe * gamma * I - (1 / self.D_hospital_lag) * Sev
        dSev_H = (1 / self.D_hospital_lag) * Sev - (1 / self.D_recovery_severe) * Sev_H
        dF = p_fatal * gamma * I - (1 / self.D_death) * F
        dR_Mild = (1 / self.D_recovery_mild) * Mild
        dR_Sev = (1 / self.D_recovery_severe) * Sev_H
        dR_F = (1 / self.D_death) * F

        return np.array([dS, dE, dI, dMild, dSev, dSev_H, dF, dR_Mild, dR_Sev, dR_F])

    def solve(self, tmin, tmax):
        self.tmin = tmin
        self.tmax = tmax
        self.sol = solve_ivp(
            self.seir_expanded,
            t_span=(self.tmin, self.tmax),
            y0=self.inital_state,
            first_step=1,
            max_step=2,
            vectorized=True,
            dense_output=True,
        )

    def plot(self, normalize=True, log_scale=False):
        t = np.arange(self.tmin, self.tmax + 1, 1)
        self.z = self.sol.sol(t)
        if not normalize:
            self.z = self.z * self.N

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # ax.plot(t, self.z.T[:, [1, 2, 5, 6]])
        ax.plot(t, self.z.T)
        # ax.axhline(y=800, color='gray', ls='--', label="__nolegend__")
        ax.axvline(x=self.D_lockdown, color="gray", ls=":", label="__nolegend__")
        ax.set_xlim(left=self.tmin, right=self.tmax)
        labels = ["S", "E", "I", "Mild", "Sev", "Sev_H", "F", "R_Mild", "R_Sev", "R_F"]
        ax.legend(
            ["Exposed", "Infectious", "Hospitalized", "Fatalities"], loc="upper right"
        )
        ax.legend(labels, ncol=2)
        ax.set_xlabel("days")
        if normalize:
            ax.ticklabel_format(useMathText=True, scilimits=(0, 0), axis="y")
        if log_scale:
            ax.set_yscale("log")
        sns.despine()
        plt.tight_layout()

    def plot_hist(self, normalize=True, log_scale=False):
        t = np.arange(self.tmin, self.tmax + 1, 1)
        self.z = self.sol.sol(t)
        if not normalize:
            self.z = self.z * self.N

        weights_exposed = cv.z[1, :]
        weights_infectious = cv.z[2, :]
        weights_hospitalized = cv.z[5, :] + cv.z[6, :]
        weights_fatal = cv.z[9]
        weights_recovered = cv.z[7] + cv.z[8]

        cmap = plt.get_cmap("tab10")
        col_exposed = cmap(1)
        col_infectious = cmap(3)
        col_hospitalized = cmap(0)
        col_fatal = cmap(5)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # ax.axhline(y=800, color="gray", ls="--", label="__nolegend__")
        ax.axvline(x=self.D_lockdown, color="gray", ls=":", label="__nolegend__")
        sns.distplot(
            t,
            bins=len(t) // 2,
            kde=False,
            ax=ax,
            hist_kws={
                "weights": weights_exposed,
                "alpha": 0.8,
                "color": col_exposed,
                "rwidth": 0.8,
                "cumulative": False,
                "label": "Exposed",
            },
        )
        sns.distplot(
            t,
            bins=len(t) // 2,
            kde=False,
            ax=ax,
            hist_kws={
                "weights": weights_infectious,
                "alpha": 0.8,
                "color": col_infectious,
                "rwidth": 0.8,
                "cumulative": False,
                "label": "Infectious",
            },
        )
        sns.distplot(
            t,
            bins=len(t) // 2,
            kde=False,
            ax=ax,
            hist_kws={
                "weights": weights_hospitalized,
                "alpha": 0.8,
                "color": col_hospitalized,
                "rwidth": 0.8,
                "cumulative": False,
                "label": "Hospitalized",
            },
        )
        sns.distplot(
            t,
            bins=len(t) // 2,
            kde=False,
            ax=ax,
            hist_kws={
                "weights": weights_fatal,
                "alpha": 0.8,
                "color": col_fatal,
                "rwidth": 0.8,
                "cumulative": False,
                "label": "Fatalities",
            },
        )
        # sns.distplot(
        #     t,
        #     bins=len(t) // 2,
        #     kde=False,
        #     ax=ax,
        #     hist_kws={
        #         "weights": weights_recovered,
        #         "alpha": 0.8,
        #         "color": col_fatal,
        #         "rwidth": 0.8,
        #         "cumulative": False,
        #         "label": "Recovered",
        #     },
        # )
        sns.despine()
        ax.set_xlabel("days")
        ax.set_xlim(left=0, right=300)
        if not normalize:
            ax.ticklabel_format(useMathText=True, scilimits=(0, 0), axis="y")
        if log_scale:
            ax.set_yscale("log")
        plt.legend(fancybox=True, framealpha=0.8)
        plt.tight_layout()

    def plot_plotly(self):
        self.t = np.arange(self.tmin, self.tmax + 1, 1)
        self.z = self.N * self.sol.sol(self.t)

        columns = [
            "Susceptible",
            "Exposed",
            "Infectious",
            "Recovering (mild)",
            "Recovering (severe at home)",
            "Hospitalized",
            "Recovering (fatal)",
            "Recovered (mild)",
            "Recovered (severe)",
            "Fatalities",
        ]
        df = pd.DataFrame(self.z.T, columns=columns)
        df.insert(0, "Time", self.t)

        counts1, bins1 = self.get_histogram(self.t, df, "Exposed")
        counts2, bins2 = self.get_histogram(self.t, df, "Infectious")
        counts3, bins3 = self.get_histogram(self.t, df, "Hospitalized")
        counts4, bins4 = self.get_histogram(self.t, df, "Fatalities")
        counts5, bins5 = self.get_histogram(self.t, df, "Recovering (fatal)")

        df_binned = pd.DataFrame(
            np.array([counts1, counts2, counts3, counts4]).T,
            columns=["Exposed", "Infectious", "Hospitalized", "Fatalities"],
        )
        df_sum = pd.DataFrame({"Total": df_binned.sum(), "Peak": df_binned.max()})
        df_sum.iloc[-1, 0] = df_sum.iloc[-1, 1]
        df_sum.iloc[-1, 1] = ""

        fig = go.Figure()
        fig.add_trace(go.Bar(x=bins1, y=counts1, opacity=0.5, name="Exposed"))
        fig.add_trace(go.Bar(x=bins2, y=counts2, opacity=0.5, name="Infectious"))
        fig.add_trace(go.Bar(x=bins3, y=counts3, opacity=0.5, name="Hospitalized"))
        fig.add_trace(go.Bar(x=bins4, y=counts4, opacity=0.5, name="Fatalities"))
        fig.update_layout(
            barmode="overlay",
            xaxis_title="days",
            yaxis_title="",
            title="Epidemic Spread Visualisation",
            template="presentation",
        )
        # fig.show()
        return (df, df_sum, fig)

    @staticmethod
    def get_histogram(x, df, column_name):
        counts, bins = np.histogram(x, bins=len(x) // 2, weights=df[column_name])
        bins = 0.5 * (bins[:-1] + bins[1:])
        counts = counts.astype(int)
        bins = bins.astype(int)
        return counts, bins


if __name__ == "__main__":
    # y0 = [Total population, # infected]
    N0 = 7e6
    I0 = 1
    y0 = np.array([N0, I0])
    # params = [R0, Rt, Tinc, Tinf, Tlock]
    params = np.array([2.2, 0.66, 5.2, 2.9, 60])

    tmin = 0
    tmax = 365

    cv = Covid(y0, params)
    cv.solve(tmin, tmax)
    # cv.plot(normalize=False, log_scale=True)
    # cv.plot_hist(normalize=False, log_scale=False)
    df, df_sum, fig = cv.plot_plotly()
