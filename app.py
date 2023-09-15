import numpy as np
from dash import dcc, Dash
from tabs import info_tab, prediction_tab, data_tab
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.UNITED])

app.layout = dbc.Container(
    [
        dcc.Interval(
            interval=500,
            id="update-interval",
            n_intervals=0
        ),
#pd.DataFrame(columns=["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]).to_json()
        dcc.Store(
            id="simulation-data",
            data=np.array([])
        ),

        dcc.Store(
            id="prediction-data",
            data=np.array([])
        ),

        dbc.Tabs(
            [
                dbc.Tab(
                    info_tab.info_tab,
                    id="data-info-tab",
                    tab_id="data-info-tab",
                    label="Welcome",

                ),

                dbc.Tab(
                    data_tab.data_tab,
                    id="data-tab",
                    tab_id="data-tab",
                    label="About Dataset",

                ),

                dbc.Tab(
                    prediction_tab.prediction_tab,
                    id="prediction-tab",
                    tab_id="prediction-tab",
                    label="Turbine Monitoring",
                )
            ]
        )
    ]
)

server = app.server
if __name__ == '__main__':
    app.run(debug=True)
