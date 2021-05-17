import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html


def init_dashboard(server):
    dash_app = dash.Dash(server=server,
                         name='Insights',
                         url_base_pathname='/dashboard/')

    return dash_app