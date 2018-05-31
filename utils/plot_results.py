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
        marker=dict(color='rgb(255,226,219)'),
        line=dict(width=0),
        fillcolor='rgba(253,112,74, 0.2)',
        fill='tonexty')

    trace = go.Scatter(
        name='Measurement',
        x=df['Episode'],
        y=df['Mean'],
        mode='lines',
        line=dict(color='rgb(253,112,74)'),
        fillcolor='rgba(253,112,74, 0.2)',
        fill='tonexty')

    lower_bound = go.Scatter(
        name='Lower Bound',
        x=df['Episode'],
        y=df['Mean']-df['Stddev'],
        marker=dict(color='rgb(255,226,219)'),
        line=dict(width=0),
        mode='lines')

    # Trace order can be important
    # with continuous error bars
    data = [lower_bound, trace, upper_bound]

    layout = go.Layout(
        yaxis=dict(title='Score'),
        title='Average Reward Over time',
        showlegend = False)

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='average_rewards_over_episodes')

if __name__ == "__main__":
    main()
