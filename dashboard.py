import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from rona import Covid

# current estimates:
# R0 = 2.8
# Rt = 0.3 - 0.7
# timeline: lockdown 16.03, first softening: 27.04
# hospital capacity: 2000 beds


N0 = 7e6
I0 = 1
y0 = np.array([N0, I0])
t_params = np.array([2.2, 0.66, 5.2, 2.9, 60, 400, 0.5 * 2.2])
c_params = np.array([32, 11.1, 28.6, 5, 0.01, 0.1])

tmin = 0
tmax = 365
cv = Covid(y0, t_params, c_params)
cv.solve_for_dashboard(tmin, tmax)
df, df_sum, fig = cv.plot_plotly()
df_sum.insert(0, "", list(df_sum.index))


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {"background": "#002b36", "text": "#7FDBFF"}
fig.update_layout(
    plot_bgcolor=colors["background"],
    paper_bgcolor=colors["background"],
    font={"color": colors["text"]},
)

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="the 'rona", style={"textAlign": "center", "color": colors["text"]}
        ),
        dcc.Graph(id="rona", figure=fig),
        html.Div(
            [
                html.H6(children="Transmission", className="app-header",),
                html.Div(
                    id="R0-slider-output-container", className="output-container",
                ),
                dcc.Slider(
                    id="R0-slider",
                    min=0,
                    max=10,
                    value=2.2,
                    marks={str(i): str(i) for i in range(0, 12, 2)},
                    step=0.1,
                ),
                html.Div(
                    id="incubation-slider-output-container",
                    className="output-container",
                ),
                dcc.Slider(
                    id="incubation-slider", min=0.15, max=24, value=5.2, step=0.1,
                ),
                html.Div(
                    id="infectious-slider-output-container",
                    className="output-container",
                ),
                dcc.Slider(id="infectious-slider", min=0, max=24, value=2.9, step=0.1,),
            ],
            className="parameter-container",
        ),
        html.Div(
            [
                html.H6(children="Lockdown", className="app-header",),
                html.Div(
                    id="lockdown-slider-output-container", className="output-container",
                ),
                dcc.Slider(
                    id="lockdown-slider",
                    min=0,
                    max=365,
                    value=60,
                    marks={str(i): str(i) for i in [0, 365]},
                    step=1,
                ),
                html.Div(
                    id="intervention-slider-output-container",
                    className="output-container",
                ),
                dcc.Slider(
                    id="intervention-slider",
                    min=0,
                    max=1,
                    value=0.66,
                    marks={str(i): str(i) for i in [0, 1]},
                    step=0.05,
                ),
                html.Div(
                    id="duration-slider-output-container", className="output-container",
                ),
                dcc.Slider(
                    id="duration-slider",
                    min=0,
                    max=365,
                    value=60,
                    marks={str(i): str(i) for i in [0, 365]},
                    step=1,
                ),
                html.Div(
                    id="R0-after-slider-output-container", className="output-container",
                ),
                dcc.Slider(
                    id="R0-after-slider",
                    min=0,
                    max=9,
                    value=2,
                    marks={str(i): str(i) for i in range(0, 12, 2)},
                    step=0.1,
                ),
            ],
            className="parameter-container",
        ),
        html.Div(
            [
                html.H6(children="Hospitalization", className="app-header",),
                html.Div(
                    id="recovery-severe-slider-output-container",
                    className="output-container",
                ),
                dcc.Slider(
                    id="recovery-severe-slider", min=1, max=100, value=29, step=1,
                ),
                html.Div(
                    id="hospital-lag-slider-output-container",
                    className="output-container",
                ),
                dcc.Slider(id="hospital-lag-slider", min=1, max=100, value=5, step=1,),
                html.Div(
                    id="p-severe-slider-output-container", className="output-container",
                ),
                dcc.Slider(id="p-severe-slider", min=0, max=1, value=0.2, step=0.01,),
            ],
            className="parameter-container",
        ),
        html.Div(
            [
                html.H6(children="Recovery & Mortality", className="app-header",),
                html.Div(
                    id="death-slider-output-container", className="output-container",
                ),
                dcc.Slider(id="death-slider", min=3, max=100, value=32, step=1,),
                html.Div(
                    id="recovery-mild-slider-output-container",
                    className="output-container",
                ),
                dcc.Slider(
                    id="recovery-mild-slider", min=1, max=100, value=11, step=1,
                ),
                html.Div(
                    id="p-fatal-slider-output-container", className="output-container",
                ),
                dcc.Slider(id="p-fatal-slider", min=0, max=0.1, value=0.01, step=0.001,),
            ],
            className="parameter-container",
        ),
    ],
)


@app.callback(
    [
        Output("rona", "figure"),
        Output("R0-slider-output-container", "children"),
        Output("incubation-slider-output-container", "children"),
        Output("infectious-slider-output-container", "children"),
        Output("lockdown-slider-output-container", "children"),
        Output("intervention-slider-output-container", "children"),
        Output("duration-slider-output-container", "children"),
        Output("R0-after-slider-output-container", "children"),
        Output("death-slider-output-container", "children"),
        Output("recovery-mild-slider-output-container", "children"),
        Output("recovery-severe-slider-output-container", "children"),
        Output("hospital-lag-slider-output-container", "children"),
        Output("p-fatal-slider-output-container", "children"),
        Output("p-severe-slider-output-container", "children"),
    ],
    [
        Input("R0-slider", "value"),
        Input("incubation-slider", "value"),
        Input("infectious-slider", "value"),
        Input("lockdown-slider", "value"),
        Input("intervention-slider", "value"),
        Input("duration-slider", "value"),
        Input("R0-after-slider", "value"),
        Input("death-slider", "value"),
        Input("recovery-mild-slider", "value"),
        Input("recovery-severe-slider", "value"),
        Input("hospital-lag-slider", "value"),
        Input("p-fatal-slider", "value"),
        Input("p-severe-slider", "value"),
    ],
)
def update_figure(
    R0,
    D_incubation,
    D_infectious,
    D_lockdown,
    OMInterventionAmt,
    D_lockdown_duration,
    R0_after,
    D_death,
    D_recovery_mild,
    D_recovery_severe,
    D_hospital_lag,
    p_fatal,
    p_severe,
):
    y0 = np.array([N0, I0])
    transmission_params = np.array(
        [
            R0,
            OMInterventionAmt,
            D_incubation,
            D_infectious,
            D_lockdown,
            D_lockdown_duration,
            R0_after,
        ]
    )
    clinical_params = np.array(
        [D_death, D_recovery_mild, D_recovery_severe, D_hospital_lag, p_fatal, p_severe]
    )

    # clinical_params = np.array([time_to_death, D_recovery_mild, D_recovery_severe, D_hospital_lag, p_fatal, p_severe])

    cv = Covid(y0, transmission_params, clinical_params)
    cv.solve_for_dashboard(tmin, tmax)
    df, df_sum, fig = cv.plot_plotly()
    # print(df_sum)

    fig.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font={"color": colors["text"]},
    )

    sliderText = "R0 = {:.1f}".format(R0)
    slider2Text = "Incubation time: {:.1f} days".format(D_incubation)
    slider3Text = "Infectious time: {:.1f} days".format(D_infectious)

    slider4Text = "lockdown active after: {:.0f} days".format(D_lockdown)
    slider5Text = "decrease transmission by: {:.2f}".format(OMInterventionAmt)
    slider6Text = "lockdown duration: {:.0f} days".format(D_lockdown_duration)
    slider7Text = "R0 after lockdown: {:.1f}".format(R0_after)

    slider8Text = "Time to death: {:.0f} days".format(D_death)
    slider9Text = "Time to recovery (mild): {:.0f} days".format(D_recovery_mild)
    slider10Text = "Time to recovery (severe): {:.0f} days".format(D_recovery_severe)
    slider11Text = "Time to hospitalization: {:.0f} days".format(D_hospital_lag)
    slider12Text = "Case fatality rate: {:.1f}%".format(100 * p_fatal)
    slider13Text = "Hospitalization rate: {:.1f}".format(100 * p_severe)

    return (
        fig,
        sliderText,
        slider2Text,
        slider3Text,
        slider4Text,
        slider5Text,
        slider6Text,
        slider7Text,
        slider8Text,
        slider9Text,
        slider10Text,
        slider11Text,
        slider12Text,
        slider13Text,
    )


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=True)
