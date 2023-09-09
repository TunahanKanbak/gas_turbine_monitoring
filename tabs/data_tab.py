import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash import html, dcc

df=pd.read_csv("static/whole_data.csv")

data_tab = dbc.Container(
    [
        dbc.Row(
            [
                html.H1("Data Dictionary"),

                dbc.Col(
                    [
                        html.P("This dataset contains 9 different operational parameters which affect gas turbine"
                               " operations. Gas turbine is an equipment where atmospheric air is compressed "
                               "and flow through combustion chamber. In the combustion chamber, air is mixed with "
                               "chosen fuel, mostly Natural Gas, and ignited. Ignition increases the chamber temperature "
                               "which leads to higher energy output from turbine. Ignited air-fuel mixture is passed "
                               "through a turbine where it expands to lower pressures while releasing its potential as"
                               "mechanical energy."),

                        html.P("During combustion process, ignition reactions don't occur in their ideal paths. Different "
                               "side products are formed due to this non-ideality. Some of these side products are "
                               "called emission such as NOx and CO etc. Since emission gas formation depends on the "
                               "combustion process and these (right side) operational parameters also affect this side "
                               "products, operational parameters can be used to predict emission gas concentraion in "
                               "exhaust gas."),

                        html.P("Right side of the page contains detail descriptions of each feature. All of these "
                               "feature are used to predict NOx emission in the exhaust gas.")
                    ],
                    align="center",
                ),

                dbc.Col(
                    [
                        dbc.Accordion(
                            [
                                dbc.AccordionItem(
                                    [
                                        html.P("This feature is recorded with unit of Celcius, C"),
                                        html.P("Ambient temperature indicates the temperature of intake air which is air "
                                               "temperature at the compressor inlet.")
                                    ],
                                    title="Ambient Temperature (AT)"
                                ),

                                dbc.AccordionItem(
                                    [
                                        html.P("This feature is recorded with unit of milibar, mbar"),
                                        html.P("Ambient pressure indicates the pressure of intake air which is air "
                                               "pressure at the compressor inlet.")
                                    ],
                                    title="Ambient Pressure (AP)"
                                ),

                                dbc.AccordionItem(
                                    [
                                        html.P("This feature is recorded with unit of percentage, %"),
                                        html.P("Ambient humidity indicates the humidity level ol intake air which is "
                                               "air humidity at the compressor inlet.")
                                    ],
                                    title="Ambient Humidity (AH)"
                                ),

                                dbc.AccordionItem(
                                    [
                                        html.P("This feature is recorded with unit of milibar, mbar"),
                                        html.P("Gas turbines usually filters intake air before it reaches compressor "
                                               "blade. Dust and other foreign particles in the air might accumulate "
                                               "on compressor/turbine blades and lead to corrosion. As time passes, this "
                                               "filter starts to clogged and differential pressure increases. This feature"
                                               " indicates this differential pressure.")
                                    ],
                                    title="Air Filter Difference Pressure (AFDP)"
                                ),

                                dbc.AccordionItem(
                                    [
                                        html.P("This feature is recorded with unit of milibar, mbar"),
                                        html.P("Ignited air-fuel mixture expands through turbine and exits the turbine "
                                               "at low pressures. Lower the pressure, higher the energy output. This "
                                               "feature indicates that exit pressure.")
                                    ],
                                    title="Gas Turbine Exhaust Pressure (GTEP)"
                                ),

                                dbc.AccordionItem(
                                    [
                                        html.P("This feature is recorded with unit of Celcius, C"),
                                        html.P("After ignition, air-fuel mixture reaches to higher temperatures, which "
                                               "affects turbine energy output directly. This feature indicates that "
                                               "temperature after ignition.")
                                    ],
                                    title="Turbine Inlet Temperature (TIT)"
                                ),

                                dbc.AccordionItem(
                                    [
                                        html.P("This feature is recorded with unit of Celcius, C"),
                                        html.P("As air-fuel mixture expands through turbine, its temperature decreases. "
                                               "This temperature is important to calculate turbine efficieny. This feature "
                                               "indicates that decreased gas temperature at turbine exit.")
                                    ],
                                    title="Turbine After Temperature (TAT)"
                                ),

                                dbc.AccordionItem(
                                    [
                                        html.P("This feature is recorded with unit of bar, bar"),
                                        html.P("Higher air pressure leads to higher energy outputs in gas turbines. "
                                               "Therefore, compressors are coupled with turbines in gas turbines. A "
                                               "compressor increases air pressure by exerting force on it. This feature "
                                               "indicates the pressure after the compression of air.")
                                    ],
                                    title="Compressor Discharge Pressure (CDP)"
                                ),

                                dbc.AccordionItem(
                                    [
                                        html.P("This feature is recorded with unit of mega-watt-hour, MWh"),
                                        html.P("This feature is direct measurement of how much energy is obtained from "
                                               "gas turbine at recorded operation window.")
                                    ],
                                    title="Turbine Energy Yield (TEY)"
                                ),

                                dbc.AccordionItem(
                                    [
                                        html.P("This feature is recorded with unit of part per million, ppm"),
                                        html.P("As gas turbine operates, side reactions always occur. Therefore, emission "
                                               "gases are formed and released to atmosphere by exhaust gas. This feature "
                                               "indicates the concentration of emission gas, NOx, in exhaust gas.")
                                    ],
                                    title="Nitrogen Oxides (NOX)"
                                )
                            ],
                            start_collapsed=True
                        )
                    ]
                )
            ],
            align="center"
        ),

        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(figure=px.violin(df, y="TEY", box=True, points="all", color_discrete_sequence=["green"])),
                    width=6
                ),

                dbc.Col(
                    dcc.Graph(figure=px.violin(df, y="NOX", box=True, points="all", color_discrete_sequence=["green"])),
                    width=6
                ),
            ],
            align="center"
        ),

        dbc.Row(
            [
                html.H1("Engineered Features"),

                dbc.Col(
                    [
                        html.P("In order to increase the predictive power of machine learning model. Extra features are "
                               "created from original data or external data. Since there is no information about "
                               "location, time or equipment model, only original data is used to create new features."),

                        html.P("Feature engineering is carried out by searching through literature to find new parameters, "
                               "which can be derived from original data. These parameters are explained below."),

                    ]
                ),

                dbc.Col(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    html.P("First of all, temperature values are converted to Kelvin, since absolute "
                                           "scale of temperature is Kelvin."),
                                    html.P("Secondly, CDP is converted to milibar to present it on the same scale with "
                                           "other pressure based features.")
                                ],
                                title="Unit Conversions"
                            ),

                            dbc.AccordionItem(
                                [
                                    html.P("As shown in the data dictionary part, there is no information about the "
                                           "temperature of air after compression. Compression process increases the "
                                           "temperature of air and affects the combustion process."),
                                    html.P("Therefore, TAC is derived from ambient and compressor discharge pressure "
                                           "by assuming gamma value as 1.4."),
                                    html.A("Detailed calculations can be seen by this link.",
                                           href="https://www.grc.nasa.gov/www/k-12/airplane/compth.html")
                                ],
                                title="Temperature After Compression (TAC)"
                            ),

                            dbc.AccordionItem(
                                [
                                    html.P("Every gas turbine is characterized by its Brayton Cycle Efficiency which "
                                           "indicates the how much work is obtained from the gas turbine relative to "
                                           "energy input of system. Brayton efficiency is calculated by four different "
                                           "temperature value which are AT, TAC, TAT, TIT. Both ideal and real "
                                           "efficiencies are calculated and an efficiency ratio is defined."),
                                    html.A("Detailed calculations can be seen by this link.",
                                           href="https://web.mit.edu/16.unified/www/SPRING/propulsion/notes/node27.html")
                                ],
                                title="Brayton Cycle Efficiency via Temperature Changes"
                            ),

                            dbc.AccordionItem(
                                [
                                    html.P("According to the literature, NOx emission has a dependency on sqaure root "
                                           "of CDP. Therefore, square root of CDP is calculated before model training."),
                                    html.A("Details about NOx emission can be seen by this link.",
                                           href="https://www.ge.com/content/dam/gepower-new/global/en_US/downloads/gas-new-site/resources/reference/ger-4211-gas-turbine-emissions-and-control.pdf")
                                ],
                                title="Square Root of CDP"
                            )
                        ],
                        start_collapsed=True
                    )
                )
            ],
            align="center"
        )
    ]
)

