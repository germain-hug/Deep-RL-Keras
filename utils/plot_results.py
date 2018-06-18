import sys
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go

fill_c = 'rgba(68, 68, 68, 0.1)'
line_c = ['rgb(31, 119, 180)', 'rgb(180, 80, 110)', 'rgb(180, 119, 31)']

def main(args=None):

    upper_bound, trace, lower_bound = [], [], []

    for i in range(len(sys.argv)-1):

        df = pd.read_csv(sys.argv[i+1])

        upper_bound.append(go.Scatter(
            name='Upper Bound',
            x=df['Episode'],
            y=df['Mean']+df['Stddev'],
            mode='lines',
            marker=dict(color="444"),
            line=dict(width=0),
            fillcolor=fill_c,
            fill='tonexty'))

        trace.append(go.Scatter(
            name='Measurement',
            x=df['Episode'],
            y=df['Mean'],
            mode='lines',
            fillcolor=fill_c,
            fill='lines'))

        lower_bound.append(go.Scatter(
            name='Lower Bound',
            x=df['Episode'],
            y=df['Mean']-df['Stddev'],
            marker=dict(color="444"),
            line=dict(width=0),
            mode='lines'))

    # Trace order can be important
    # with continuous error bars
    # data = [*lower_bound, *trace, *upper_bound]
    data = trace

    layout = go.Layout(
        yaxis=dict(title='Score'),
        title='Average Reward',
        showlegend = True)

    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='average-reward')

if __name__ == "__main__":
    main()
