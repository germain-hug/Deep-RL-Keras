import sys
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go

def main(args=None):

    df = pd.read_csv(sys.argv[1])

    upper_bound = go.Scatter(
        name='Upper Bound',
        x=df['Episode'],
        y=df['Mean']+df['Stddev'],
        mode='lines',
        marker=dict(color="444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    trace = go.Scatter(
        name='Measurement',
        x=df['Episode'],
        y=df['Mean'],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    lower_bound = go.Scatter(
        name='Lower Bound',
        x=df['Episode'],
        y=df['Mean']-df['Stddev'],
        marker=dict(color="444"),
        line=dict(width=0),
        mode='lines')

    # Trace order can be important
    # with continuous error bars
    data = [lower_bound, trace, upper_bound]

    layout = go.Layout(
        yaxis=dict(title='Wind speed (m/s)'),
        title='Continuous, variable value error bars.<br>Notice the hover text!',
        showlegend = False)

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='pandas-continuous-error-bars')

if __name__ == "__main__":
    main()
