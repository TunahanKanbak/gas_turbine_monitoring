import time
import dash_bootstrap_components as dbc
from prediction import predictor
import numpy as np
import pandas as pd
from dash import dcc, Output, Input, State, callback, no_update
from dash.exceptions import PreventUpdate
import plotly.express as px
import datetime
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = 'simple_white'

prediction_tab = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label(
                            "Stop/Start Monitoring",
                            style={"font-weight": "bold", "font-size": "10px"}
                        ),
                        dbc.Switch(
                            id="stop-monitoring",
                            value=False
                        )
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Ambient Temperature (C)", style={"font-weight": "bold", "font-size": "20px"}),
                        dcc.Graph(id="att-at")
                    ],
                    width=4
                ),

                dbc.Col(
                    [
                        dbc.Label("Ambient Pressure (mbar)", style={"font-weight": "bold", "font-size": "20px"}),
                        dcc.Graph(id="att-ap")
                    ],
                    width=4
                ),

                dbc.Col(
                    [
                        dbc.Label("Ambient Humidity (%)", style={"font-weight": "bold", "font-size": "20px"}),
                        dcc.Graph(id="att-ah")
                    ],
                    width=4
                ),

                dbc.Col(
                    [
                        dbc.Label("Air Filter Differential Pressure (mbar)",
                                  style={"font-weight": "bold", "font-size": "20px"}),
                        dcc.Graph(id="att-afdp")
                    ],
                    width=4
                ),

                dbc.Col(
                    [
                        dbc.Label("Gas Turbine Exhaust Pressure (mbar)",
                                  style={"font-weight": "bold", "font-size": "20px"}),
                        dcc.Graph(id="att-gtep")
                    ],
                    width=4
                ),

                dbc.Col(
                    [
                        dbc.Label("Turbine Inlet Temperature (C)",
                                  style={"font-weight": "bold", "font-size": "20px"}),
                        dcc.Graph(id="att-tit")
                    ],
                    width=4
                ),

                dbc.Col(
                    [
                        dbc.Label("Turbine After Temperature (C)",
                                  style={"font-weight": "bold", "font-size": "20px"}),
                        dcc.Graph(id="att-tat")
                    ],
                    width=4
                ),

                dbc.Col(
                    [
                        dbc.Label("Turbine Energy Yield (MWh)",
                                  style={"font-weight": "bold", "font-size": "20px"}),
                        dcc.Graph(id="att-tey")
                    ],
                    width=4
                ),

                dbc.Col(
                    [
                        dbc.Label("Compressor Discharge Pressure (bar)",
                                  style={"font-weight": "bold", "font-size": "20px"}),
                        dcc.Graph(id="att-cdp")
                    ],
                    width=4
                )
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("NOx Prediction (ppm)", style={"font-weight": "bold", "font-size": "25px"}),
                        dcc.Graph(id="prediction")
                    ],
                    width=9
                )
            ],
            justify="center"
        )
    ]
)

@callback(
    Output("simulation-data", "data"),
    Output("prediction-data", "data"),
    Input("update-interval", "n_intervals"),
    State("simulation-data", "data"),
    State("prediction-data", "data"),
    State("stop-monitoring", "value"),
    prevent_initial_call=True
)
def synthetize_data(n_int, data, predictions, monitoring):
    if monitoring:
        data = np.array(data)
        predictions = np.array(predictions)
        beta = 0.9
        last_row = data.shape[0]
        if last_row == 0:
            set_seed(2023)

        synthetic_list = [
            np.random.rand() * (40 - -7) - 7,
            np.random.rand() * (1040 - 980) + 980,
            np.random.rand() * (101 - 24) + 24,
            np.random.rand() * (8 - 2) + 2,
            np.random.rand() * (41 - 17) + 17,
            np.random.rand() * (1100 - 1000) + 1000,
            np.random.rand() * (551 - 511) + 511,
            np.random.rand() * (180 - 100) + 100,
            np.random.rand() * (16 - 9) + 9
        ]

        if last_row == 0:
            data = np.array([synthetic_list])
        else:
            last_data_list = data[-1]
            new_line = [[(beta * last_data + (1 - beta) * new_data)
                         for last_data, new_data
                         in zip(last_data_list, synthetic_list)]]
            data = np.append(data, new_line, axis=0)

        prediction = predictor(data[-1])[0]

        if last_row == 0:
            predictions = np.array([[prediction, prediction + np.random.rand() * 7.2 - 3.6]])
        else:
            predictions = np.append(predictions, [[prediction, prediction + np.random.rand() * 7.2 - 3.6]], axis=0)
        return data, predictions
    else:
        np.save("static/data", data)
        np.save("static/pred", predictions)
        raise PreventUpdate

def set_seed(seed):
    np.random.seed(seed)

@callback(
    Output("att-at", "figure"),
    Output("att-ap", "figure"),
    Output("att-ah", "figure"),
    Output("att-afdp", "figure"),
    Output("att-gtep", "figure"),
    Output("att-tit", "figure"),
    Output("att-tat", "figure"),
    Output("att-tey", "figure"),
    Output("att-cdp", "figure"),
    Output("prediction", "figure"),
    Input("simulation-data", "data"),
    State("prediction-data", "data"),
    prevent_initial_call=True
)
def construct_graphs(sim, pred):
    sim = np.array(sim)
    pred = np.array(pred)

    base = datetime.datetime.today()
    date_list = [base + datetime.timedelta(hours=x) for x in range(sim.shape[0])]

    att_1_fig = px.line(x=date_list, y=sim[:, 0], labels={"x": "Time", "y": "Temperature"})
    att_2_fig = px.line(x=date_list, y=sim[:, 1], labels={"x": "Time", "y": "Pressure"})
    att_3_fig = px.line(x=date_list, y=sim[:, 2], labels={"x": "Time", "y": "Humidity"})
    att_4_fig = px.line(x=date_list, y=sim[:, 3], labels={"x": "Time", "y": "Pressure"})
    att_5_fig = px.line(x=date_list, y=sim[:, 4], labels={"x": "Time", "y": "Pressure"})
    att_6_fig = px.line(x=date_list, y=sim[:, 5], labels={"x": "Time", "y": "Temperature"})
    att_7_fig = px.line(x=date_list, y=sim[:, 6], labels={"x": "Time", "y": "Temperature"})
    att_8_fig = px.line(x=date_list, y=sim[:, 7], labels={"x": "Time", "y": "Energy"})
    att_9_fig = px.line(x=date_list, y=sim[:, 8], labels={"x": "Time", "y": "Pressure"})
    pred_fig = px.line()
    pred_fig.add_trace(go.Scatter(x=date_list, y=pred[:, 0], name="Prediction", mode="lines"))
    pred_fig.add_trace(go.Scatter(x=date_list, y=pred[:, 1], name="Real", mode="lines"))

    return att_1_fig, att_2_fig, att_3_fig,\
        att_4_fig, att_5_fig, att_6_fig,\
        att_7_fig, att_8_fig, att_9_fig,\
        pred_fig