import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()

x = ['2010', '2011', '2012', '2013', '2014', '2014', '2015', '2016', '2017', 'Human']
y = [28.2, 25.8, 16.2, 11.7, 7.3, 6.7, 3.6, 3.0, 2.3, 5.1]
#x = ['shallow', 'shallow', '8 layers', '8 layers', '19', '22', '152', '152', '152']
y2 = [None, None, 8, 8, 19, 22, 152, 152, 152]

trace1 = go.Bar(x=x, y=y, text=y, textposition='outside', opacity=0.8, name='Error Percentage')
trace2 = go.Scatter(x=x, y=y2, yaxis='y2', name='Number of Layers')

layout = go.Layout(title='ImageNet Large Scale Visual Recognition Challenge (ILSVRC) Winners',
                   xaxis={'type': 'category'},
                   yaxis2={'overlaying': 'y', 'side': 'right', 'zeroline': False})
fig = go.Figure(data=[trace1, trace2], layout=layout)
py.iplot(fig, show_link=False)
