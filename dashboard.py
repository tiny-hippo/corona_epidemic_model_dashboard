import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from rona import Covid


N0 = 7e6
I0 = 1
y0 = np.array([N0, I0])
params = np.array([2.2, 0.66, 5.2, 2.9, 60, 400, 1.1])
tmin = 0
tmax = 365
cv = Covid(y0, params)
cv.solve(tmin, tmax)
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
                html.H5(
                    children="Transmission Dynamics",
                    style={"textAlign": "center", "color": colors["text"]},
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
                    id="R0-slider-output-container",
                    style={"textAlign": "center", "color": colors["text"]},
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
                    id="lockdown-slider-output-container",
                    style={"textAlign": "center", "color": colors["text"]},
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
                    id="intervention-slider-output-container",
                    style={"textAlign": "center", "color": colors["text"]},
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
                    id="duration-slider-output-container",
                    style={"textAlign": "center", "color": colors["text"]},
                ),
                dcc.Slider(
                    id="R0-after-slider",
                    min=0,
                    max=9,
                    value=2,
                    marks={str(i): str(i) for i in range(0, 12, 2)},
                    step=0.1,
                ),
                html.Div(
                    id="R0-after-slider-output-container",
                    style={"textAlign": "center", "color": colors["text"]},
                ),
            ],
            style={"width": "20%", "display": "inline-block", "padding": "0 20"},
        ),
        html.Div(
            [
                html.H5(
                    children="Clinical Dynamics",
                    style={"textAlign": "center", "color": colors["text"]},
                ),
                dcc.Slider(
                    id="test-slider",
                    min=1,
                    max=100,
                    value=50,
                    marks={str(i): str(i) for i in range(1, 101, 10)},
                    step=10,
                ),
            ],
            style={
                "width": "20%",
                "display": "inline-block",
                "padding": "0 20",
                "verticalAlign": "top",
            },
        ),
    ],
)


@app.callback(
    [
        Output("rona", "figure"),
        Output("R0-slider-output-container", "children"),
        Output("lockdown-slider-output-container", "children"),
        Output("intervention-slider-output-container", "children"),
        Output("duration-slider-output-container", "children"),
        Output("R0-after-slider-output-container", "children"),
    ],
    [
        Input("R0-slider", "value"),
        Input("lockdown-slider", "value"),
        Input("intervention-slider", "value"),
        Input("duration-slider", "value"),
        Input("R0-after-slider", "value"),
    ],
)
def update_figure(R0, D_lockdown, OMInterventionAmt, D_lockdown_duration, R0_after):
    y0 = np.array([N0, I0])
    transmission_params = np.array(
        [R0, OMInterventionAmt, 5.2, 2.9, D_lockdown, D_lockdown_duration, R0_after]
    )
    # clinical_params = np.array([time_to_death, D_recovery_mild, D_recovery_severe, D_hospital_lag, cfr, p_severe])

    cv = Covid(y0, transmission_params)
    cv.solve(tmin, tmax)
    df, df_sum, fig = cv.plot_plotly()
    print(df_sum)

    fig.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font={"color": colors["text"]},
    )

    sliderText = "R0 = {:.1f}".format(R0)
    slider2Text = "lockdown active after {:.0f} days".format(D_lockdown)
    slider3Text = "decrease transmission by {:.2f}".format(OMInterventionAmt)
    slider4Text = "lockdown duration {:.0f} days".format(D_lockdown_duration)
    slider5Text = "R0 after lockdown: {:.1f}".format(R0_after)

    return (fig, sliderText, slider2Text, slider3Text, slider4Text, slider5Text)


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=True)
