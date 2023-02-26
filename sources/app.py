import dash
from dash_extensions.enrich import DashProxy, MultiplexerTransform

app = dash.Dash(__name__, suppress_callback_exceptions=False, prevent_initial_callbacks=True)
server = app.server