import dash_core_components as dcc
import dash_html_components as html
import dash

from app import app
from app import server
from layouts import tab1, tab2, tab3
import callbacks

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content')
])

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    print("pathname: ", pathname)
    if pathname == '/apps/tab1':
        return tab1
    elif pathname == '/apps/tab2':
        return tab2
    elif pathname == '/apps/tab3':
        return tab3
    else:
        return tab3 # This is the "file upload page"

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
