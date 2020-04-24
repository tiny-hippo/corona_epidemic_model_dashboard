import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from rona import Covid


N0 = 7e6
I0 = 1
y0 = np.array([N0, I0])
params = np.array([2.2, 0.66, 5.2, 2.9, 60])
tmin = 0
tmax = 365
cv = Covid(y0, params)
cv.solve(tmin, tmax)
df, df_sum, fig = cv.plot_plotly()
df_sum.insert(0, "", list(df_sum.index))


def generate_table(dataframe, max_rows=10):
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in dataframe.columns])),
            html.Tbody(
                [
                    html.Tr(
                        [html.Td(dataframe.iloc[i][col]) for col in dataframe.columns]
                    )
                    for i in range(min(len(dataframe), max_rows))
                ]
            ),
        ],
        style={"color": colors["text"]},
    )


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
            children="The 'rona", style={"textAlign": "center", "color": colors["text"]}
        ),
        html.Div(
            children="built with Dash: A web application framework  for Python.",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        dcc.Graph(id="rona", figure=fig,),
        html.Div(
            [
                dcc.Slider(
                    id="R0-slider",
                    min=1,
                    max=9,
                    value=2,
                    marks={str(i): str(i) for i in range(1, 10, 2)},
                    step=0.1,
                ),
                html.Div(
                    id="slider-output-container",
                    style={"textAlign": "center", "color": colors["text"]},
                ),
                dcc.Slider(
                    id="intervention-slider",
                    min=0,
                    max=1,
                    value=0.66,
                    marks={str(i): str(i) for i in [0, 1]},
                    step=0.1,
                ),
                html.Div(
                    id="slider2-output-container",
                    style={"textAlign": "center", "color": colors["text"]},
                ),
            ],
            style={"width": "20%", "display": "inline-block", "padding": "0 20"},
        ),
        html.H4(children="Summary", style={"color": colors["text"]}),
        generate_table(df_sum),
    ],
)


@app.callback(
    [
        Output("rona", "figure"),
        Output("slider-output-container", "children"),
        Output("slider2-output-container", "children"),
    ],
    [Input("R0-slider", "value"), Input("intervention-slider", "value")],
)
def update_figure(R0, OMInterventionAmt):
    y0 = np.array([N0, I0])
    params = np.array([R0, OMInterventionAmt, 5.2, 2.9, 60])
    cv = Covid(y0, params)
    cv.solve(tmin, tmax)
    df, df_sum, fig = cv.plot_plotly()

    fig.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font={"color": colors["text"]},
    )

    sliderText = "R0 = {:.1f}".format(R0)
    slider2Text = "Intervention: decrease transmission by {:.0f}%".format(100 * OMInterventionAmt)
    return (fig, sliderText, slider2Text)


if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=True)
