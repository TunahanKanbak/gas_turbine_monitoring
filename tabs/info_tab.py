from dash import html
import dash_bootstrap_components as dbc

info_tab = dbc.Container(
    [
        dbc.Row(
            [
                html.H3("Welcome to the Turbine Exhaust NOx Emission Monitoring System!")
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P("This web-app is demo application of a NOx emission prediction model. Model "
                               "uses a Histogram Gradient Boosting Machine which is a small, fast but high performance "
                               "tree based model."),

                        html.P("Training dataset is retrieved from UCI Machine Learning Repository. This dataset is "
                               "collected from a energy production facility in Turkey by some academic researcher."),

                        html.P("In the next page, a simulation of gas turbine operation is shown by various operation "
                               "parameters such as ambient temperature, turbine inlet temperature, turbine energy "
                               "yield etc."),

                        html.P("At the bottom, you can see the predicted value for NOx emission and "
                               "its comparison to real NOx emission which is derived from the prediction."),

                        html.P("Since plots are constantly refreshed, data generation should be stopped to perform "
                               "detail inspection on graphs. In order to stop the generation, you can use the switch "
                               "button at top of the page"),

                        html.A("You can reach to dataset by this link.",
                               href="https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set")
                    ],
                    width=6
                ),

                dbc.Col(
                    [
                        html.Br(),
                        html.Br(),
                        html.Img(
                            src="https://energyeducation.ca/wiki/images/thumb/3/3d/Natgasturb.png/800px-Natgasturb.png"
                        ),
                    ],
                    width=6
                )
            ]
        )
    ]
)