import dash
import dash_core_components as dcc
import dash_html_components as html
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
        style={"color": colors["text"]}
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
        dcc.Graph(id="Enhanced SEIR model for the spread of corona", figure=fig,),
        html.H4(children="Summary", style={"color": colors["text"]}),
        generate_table(df_sum),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True, dev_tools_hot_reload=True)
