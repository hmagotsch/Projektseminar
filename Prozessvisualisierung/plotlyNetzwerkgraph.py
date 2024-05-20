#Es wird aus einer XES-Datei ein Plotly-Netzwerkgraph erstellt
import networkx as nx
import plotly.graph_objs as go
import pm4py
import pandas as pd
import json
import plotly.io as pio
import random
import re
import pydot

log = pm4py.read_xes('C:/Users/Hannes/Desktop/Uni/Master/3.Semester/Projektseminar/TestPy/seq1and2MachineB-exported.xes')
log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], format='mixed', errors='coerce')
process_tree = pm4py.discover_process_tree_inductive(log)
bpmn_model = pm4py.convert_to_bpmn(process_tree)

dic = bpmn_model.__dict__
#print(dic)

#dict infos lesbar machen
bpmn_flows_str = str(dic.get('_BPMN__flows', set()))
bpmn_flows_str = bpmn_flows_str.replace('}', '')
bpmn_flows_str = bpmn_flows_str.replace(', id', '; id')
#print(bpmn_flows_str)
flow_data = [flow.split(' -> ') for flow in bpmn_flows_str.split(';')]
flow_edges = ['['+'; '.join([id.split('@')[-1] for id in sublist]) + ']' for sublist in flow_data]
#print(flow_edges)

# Erstellen NetworkX-Graphen
G = nx.MultiDiGraph()
for edge_str in flow_edges:
    edge_str = edge_str.strip('[]')  # [] Entfernen 
    source, target = edge_str.split('; ') 
    G.add_edge(source, target)
#G.add_edge('A', 'B')
#G.add_edge('B', 'C')
#G.add_edge('C', 'D')

#strt_pos = {'start': (-1, 0),  'end': (1, 0)}
#dist = {('Rolloeinfahrt auf Vorposition', 'Initiatior Vorposition'): 0.1}

#pos_ges = nx.spring_layout(G, seed=42, pos={'start': [0, 0]})
#pos_ges=nx.nx_pydot.graphviz_layout(G, prog='dot', root='start')
#pos_ges = nx.kamada_kawai_layout(G)
pos_ges=nx.spectral_layout(G, weight=None, scale=2, center=None, dim=2)
#pos_ges=nx.planar_layout(G, scale=1, center=None, dim=2)
#pos_ges = nx.spring_layout(G, seed=42, pos=pos_ges)
#pos_ges = nx.kamada_kawai_layout(G, pos=pos_ges)
#pos_ges = nx.kamada_kawai_layout(G)
#while pos_ges['start'][0] > pos_ges['end'][0]:
#    pos_ges = nx.kamada_kawai_layout(G)


#Knoten zufälig verschieben, da bei Kmada Kaai Layout Knoten übereinander liegen können
for node in pos_ges:
   pos_ges[node] = (pos_ges[node][0] + random.uniform(-0.1, 0.1), pos_ges[node][1] + random.uniform(-0.1, 0.1))

#pos_ges = nx.spring_layout(G, seed=42, pos=pos_ges) 
pos_ges = nx.kamada_kawai_layout(G, pos=pos_ges)  

#Verbindungen zwischen den einzelnen Knoten
edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='grey'))
for edge in G.edges():
    x0, y0 = pos_ges[edge[0]]
    x1, y1 = pos_ges[edge[1]]
    edge_trace['x'] += (x0, x1, None)
    edge_trace['y'] += (y0, y1, None)

#Knoten erstellen
node_trace = go.Scatter(x=[], y=[], mode='markers', marker=dict(size=15, color='red'))
node_labels = list(G.nodes())  #  Namen der Knoten

abstand = 0.06  # Abstand damit namen nicht auf Knoten stehen

for node in G.nodes():
    x, y = pos_ges[node]
    node_trace['x'] += (x,)
    node_trace['y'] += (y,)

node_colors = ['green' if node == 'start' else 'blue' if node == 'end' else 'red' for node in G.nodes]
node_trace.marker.color = node_colors    

fig = go.Figure(data=[edge_trace, node_trace])

#Bezeichnungen der einzelnen Knoten
font_size = 16 
#for i, label in enumerate(node_labels):
#    fig.add_annotation(x=pos_ges[label][0], y=pos_ges[label][1] + abstand,
#                       text=label, showarrow=False, font=dict(size=font_size))

#Pfeile zu den einzelnen Verbindungen    
for edge in G.edges():
    x0, y0 = pos_ges[edge[0]]
    x1, y1 = pos_ges[edge[1]]
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref='x', yref='y', axref='x', ayref='y',
                      showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5)

fig.show()


#In das richtige Ausgabeformat bringen

#fig_json = json.loads(fig.to_json())
#print (fig_json)
#fig2={'data': [{'line': {'color': 'grey', 'width': 0.5}, 'x': [0.17336554016944322, 0.2072487138593338, None, 0.17336554016944322, 0.30285228475452236, None, 0.2072487138593338, 0.12533640221734543, None, 0.2072487138593338, 0.3681749928959967, None, 0.12533640221734543, 0.1345976504168486, None, -0.0004792622289102183, 0.17336554016944322, None, 0.1345976504168486, 0.258044206084872, None, -0.5358132139862177, -0.38751610246991464, None, -0.38751610246991464, -0.16732325338912002, None, -0.16732325338912002, -0.0004792622289102183, None, 0.258044206084872, 0.6126297057157745, None, -0.6146493715047701, -0.38751610246991464, None, -1.0144723242694844, -0.8627565017528479, None, -0.8627565017528479, -0.6146493715047701, None, -0.8627565017528479, -0.5358132139862177, None, 0.5543089335815549, 0.8473938710056763, None, 0.6126297057157745, 0.5543089335815549, None, 0.3681749928959967, 0.5543089335815549, None, 0.30285228475452236, 0.1637870091026683, None, 0.1637870091026683, -0.16732325338912002, None], 'y': [0.06676142910376985, -0.19796864073377748, None, 0.06676142910376985, 0.47079265818833915, None, -0.19796864073377748, -0.45119936393334426, None, -0.19796864073377748, -0.3768978829172026, None, -0.45119936393334426, -0.544401415857865, None, 0.24023978289332146, 0.06676142910376985, None, -0.544401415857865, -0.7494618730467777, None, 0.5379539613624407, 0.3370999020682739, None, 0.3370999020682739, 0.4350074752713983, None, 0.4350074752713983, 0.24023978289332146, None, -0.7494618730467777, -0.6854975613843457, None, 0.38691359198256975, 0.3370999020682739, None, 0.5606801216598708, 
#0.610002960468695, None, 0.610002960468695, 0.38691359198256975, None, 0.610002960468695, 0.5379539613624407, None, -0.51238298813099, -0.41185579759893176, None, -0.6854975613843457, -0.51238298813099, None, -0.3768978829172026, -0.51238298813099, None, 0.47079265818833915, 0.6221171703544651, None, 0.6221171703544651, 0.4350074752713983, None], 'type': 'scatter'}, {'marker': {'size': 15}, 'mode': 'markers', 'x': [0.17336554016944322, 0.2072487138593338, 0.12533640221734543, -0.0004792622289102183, 0.1345976504168486, -0.5358132139862177, -0.38751610246991464, -0.16732325338912002, 0.258044206084872, -0.6146493715047701, -1.0144723242694844, -0.8627565017528479, 0.5543089335815549, 0.8473938710056763, 0.6126297057157745, 0.3681749928959967, 0.30285228475452236, 0.1637870091026683], 'y': [0.06676142910376985, -0.19796864073377748, -0.45119936393334426, 0.24023978289332146, -0.544401415857865, 0.5379539613624407, 0.3370999020682739, 0.4350074752713983, -0.7494618730467777, 0.38691359198256975, 0.5606801216598708, 0.610002960468695, -0.51238298813099, -0.41185579759893176, -0.6854975613843457, -0.3768978829172026, 0.47079265818833915, 0.6221171703544651], 'type': 'scatter'}], 'layout': {'template': {'data': {'histogram2dcontour': [{'type': 'histogram2dcontour', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0.0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, 
#'#fdca26'], [1.0, '#f0f921']]}], 'choropleth': [{'type': 'choropleth', 'colorbar': {'outlinewidth': 0, 'ticks': ''}}], 'histogram2d': [{'type': 'histogram2d', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0.0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, 
#'#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1.0, '#f0f921']]}], 'heatmap': [{'type': 'heatmap', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0.0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1.0, '#f0f921']]}], 'heatmapgl': [{'type': 'heatmapgl', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0.0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, 
#'#fb9f3a'], [0.8888888888888888, '#fdca26'], [1.0, '#f0f921']]}], 'contourcarpet': [{'type': 'contourcarpet', 'colorbar': {'outlinewidth': 0, 'ticks': ''}}], 'contour': [{'type': 
#'contour', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0.0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1.0, '#f0f921']]}], 'surface': [{'type': 'surface', 'colorbar': {'outlinewidth': 0, 'ticks': ''}, 'colorscale': [[0.0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1.0, '#f0f921']]}], 'mesh3d': [{'type': 'mesh3d', 'colorbar': {'outlinewidth': 0, 'ticks': ''}}], 'scatter': [{'fillpattern': {'fillmode': 'overlay', 'size': 10, 'solidity': 0.2}, 'type': 'scatter'}], 'parcoords': [{'type': 'parcoords', 'line': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scatterpolargl': [{'type': 
#'scatterpolargl', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'bar': [{'error_x': {'color': '#2a3f5f'}, 'error_y': {'color': '#2a3f5f'}, 'marker': {'line': {'color': '#E5ECF6', 'width': 0.5}, 'pattern': {'fillmode': 'overlay', 'size': 10, 'solidity': 0.2}}, 'type': 'bar'}], 'scattergeo': [{'type': 'scattergeo', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scatterpolar': [{'type': 'scatterpolar', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'histogram': [{'marker': {'pattern': {'fillmode': 'overlay', 'size': 10, 'solidity': 0.2}}, 'type': 'histogram'}], 'scattergl': [{'type': 'scattergl', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scatter3d': [{'type': 'scatter3d', 'line': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}, 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scattermapbox': [{'type': 'scattermapbox', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scatterternary': [{'type': 'scatterternary', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'scattercarpet': [{'type': 'scattercarpet', 'marker': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}}], 'carpet': [{'aaxis': {'endlinecolor': '#2a3f5f', 'gridcolor': 'white', 'linecolor': 'white', 'minorgridcolor': 'white', 'startlinecolor': '#2a3f5f'}, 'baxis': {'endlinecolor': '#2a3f5f', 'gridcolor': 'white', 'linecolor': 'white', 'minorgridcolor': 'white', 'startlinecolor': '#2a3f5f'}, 'type': 'carpet'}], 'table': [{'cells': {'fill': {'color': '#EBF0F8'}, 'line': {'color': 'white'}}, 'header': {'fill': {'color': '#C8D4E3'}, 'line': {'color': 'white'}}, 'type': 'table'}], 'barpolar': [{'marker': {'line': {'color': '#E5ECF6', 'width': 0.5}, 'pattern': {'fillmode': 'overlay', 'size': 10, 'solidity': 0.2}}, 'type': 'barpolar'}], 'pie': [{'automargin': True, 'type': 'pie'}]}, 'layout': {'autotypenumbers': 'strict', 'colorway': ['#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A', '#19d3f3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'], 'font': {'color': '#2a3f5f'}, 'hovermode': 'closest', 'hoverlabel': {'align': 'left'}, 'paper_bgcolor': 'white', 'plot_bgcolor': '#E5ECF6', 'polar': {'bgcolor': '#E5ECF6', 'angularaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}, 'radialaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': 
#''}}, 'ternary': {'bgcolor': '#E5ECF6', 'aaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}, 'baxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}, 'caxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': ''}}, 'coloraxis': {'colorbar': {'outlinewidth': 0, 'ticks': ''}}, 'colorscale': {'sequential': [[0.0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1.0, '#f0f921']], 'sequentialminus': [[0.0, '#0d0887'], [0.1111111111111111, '#46039f'], [0.2222222222222222, '#7201a8'], [0.3333333333333333, '#9c179e'], [0.4444444444444444, '#bd3786'], [0.5555555555555556, '#d8576b'], [0.6666666666666666, '#ed7953'], [0.7777777777777778, '#fb9f3a'], [0.8888888888888888, '#fdca26'], [1.0, '#f0f921']], 'diverging': [[0, '#8e0152'], [0.1, '#c51b7d'], [0.2, '#de77ae'], [0.3, '#f1b6da'], [0.4, '#fde0ef'], [0.5, '#f7f7f7'], [0.6, '#e6f5d0'], [0.7, '#b8e186'], [0.8, '#7fbc41'], [0.9, '#4d9221'], [1, '#276419']]}, 'xaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': '', 'title': {'standoff': 15}, 'zerolinecolor': 'white', 'automargin': True, 'zerolinewidth': 2}, 'yaxis': {'gridcolor': 'white', 'linecolor': 'white', 'ticks': '', 'title': {'standoff': 15}, 'zerolinecolor': 'white', 'automargin': True, 'zerolinewidth': 2}, 'scene': {'xaxis': {'backgroundcolor': '#E5ECF6', 'gridcolor': 'white', 'linecolor': 'white', 'showbackground': True, 'ticks': '', 'zerolinecolor': 'white', 'gridwidth': 2}, 'yaxis': {'backgroundcolor': '#E5ECF6', 'gridcolor': 'white', 'linecolor': 'white', 'showbackground': True, 'ticks': '', 'zerolinecolor': 'white', 'gridwidth': 2}, 'zaxis': {'backgroundcolor': '#E5ECF6', 'gridcolor': 'white', 'linecolor': 'white', 'showbackground': True, 'ticks': '', 'zerolinecolor': 'white', 'gridwidth': 2}}, 'shapedefaults': {'line': {'color': '#2a3f5f'}}, 'annotationdefaults': {'arrowcolor': '#2a3f5f', 'arrowhead': 0, 'arrowwidth': 1}, 'geo': {'bgcolor': 'white', 'landcolor': '#E5ECF6', 'subunitcolor': 'white', 'showland': True, 'showlakes': True, 'lakecolor': 'white'}, 'title': {'x': 0.05}, 'mapbox': {'style': 'light'}}}, 'annotations': [{'font': {'size': 16}, 'showarrow': False, 'text': 'xor_2_split', 'x': 0.17336554016944322, 'y': 0.12676142910376986}, {'font': {'size': 16}, 'showarrow': False, 'text': 'xor_3_split', 'x': 0.2072487138593338, 'y': -0.1379686407337775}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Bogen zu weit Seite 1', 'x': 0.12533640221734543, 'y': -0.39119936393334426}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Zähler und Maschinengeschwindigkeit', 'x': -0.0004792622289102183, 'y': 0.3002397828933214}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Schnellhalt Maschine', 'x': 0.1345976504168486, 'y': -0.48440141585786495}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Rolloeinfahrt auf Vorposition', 'x': -0.5358132139862177, 'y': 0.5979539613624407}, {'font': {'size': 16}, 'showarrow': False, 'text': 'xor_1_join', 'x': -0.38751610246991464, 'y': 0.3970999020682739}, {'font': {'size': 16}, 'showarrow': False, 'text': 'xor_2_join', 'x': -0.16732325338912002, 'y': 0.4950074752713983}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Reserve j21200_w27_b03', 'x': 0.258044206084872, 'y': -0.6894618730467776}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Initiator Vorposition', 'x': -0.6146493715047701, 'y': 0.44691359198256975}, {'font': {'size': 16}, 'showarrow': False, 'text': 'start', 'x': -1.0144723242694844, 'y': 0.6206801216598707}, {'font': {'size': 16}, 'showarrow': False, 'text': 'xor_1_split', 'x': -0.8627565017528479, 'y': 0.670002960468695}, {'font': {'size': 16}, 'showarrow': False, 'text': 'xor_3_join', 'x': 0.5543089335815549, 'y': -0.45238298813099004}, {'font': {'size': 16}, 'showarrow': False, 'text': 'end', 'x': 0.8473938710056763, 'y': -0.35185579759893176}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Nachlauf Fortdruck', 'x': 0.6126297057157745, 'y': -0.6254975613843456}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Aufforderung Überschießsperre quittieren', 'x': 0.3681749928959967, 'y': -0.3168978829172026}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Stapel frei bei Nonstop', 'x': 0.30285228475452236, 'y': 0.5307926581883391}, {'font': {'size': 16}, 'showarrow': False, 'text': 'Schutz Feuchtwerk geöffnet', 'x': 0.1637870091026683, 'y': 0.6821171703544651}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.17336554016944322, 'axref': 'x', 'ay': 0.06676142910376985, 'ayref': 'y', 'showarrow': True, 'x': 0.2072487138593338, 'xref': 'x', 'y': -0.19796864073377748, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.17336554016944322, 'axref': 'x', 'ay': 0.06676142910376985, 'ayref': 'y', 'showarrow': True, 'x': 0.30285228475452236, 'xref': 'x', 'y': 0.47079265818833915, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.2072487138593338, 'axref': 'x', 'ay': -0.19796864073377748, 'ayref': 'y', 'showarrow': True, 'x': 0.12533640221734543, 'xref': 'x', 'y': -0.45119936393334426, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.2072487138593338, 'axref': 'x', 'ay': -0.19796864073377748, 'ayref': 'y', 'showarrow': True, 'x': 0.3681749928959967, 'xref': 'x', 'y': -0.3768978829172026, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.12533640221734543, 'axref': 'x', 'ay': -0.45119936393334426, 'ayref': 'y', 'showarrow': True, 'x': 0.1345976504168486, 'xref': 'x', 'y': -0.544401415857865, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': -0.0004792622289102183, 'axref': 'x', 'ay': 0.24023978289332146, 'ayref': 'y', 'showarrow': True, 
#'x': 0.17336554016944322, 'xref': 'x', 'y': 0.06676142910376985, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.1345976504168486, 'axref': 'x', 'ay': -0.544401415857865, 'ayref': 'y', 'showarrow': True, 'x': 0.258044206084872, 'xref': 'x', 'y': -0.7494618730467777, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': -0.5358132139862177, 'axref': 'x', 'ay': 0.5379539613624407, 'ayref': 'y', 'showarrow': True, 'x': -0.38751610246991464, 'xref': 'x', 'y': 0.3370999020682739, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': -0.38751610246991464, 'axref': 'x', 'ay': 0.3370999020682739, 'ayref': 'y', 'showarrow': True, 'x': -0.16732325338912002, 'xref': 'x', 'y': 0.4350074752713983, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': -0.16732325338912002, 'axref': 'x', 'ay': 0.4350074752713983, 'ayref': 'y', 'showarrow': True, 'x': -0.0004792622289102183, 'xref': 'x', 'y': 0.24023978289332146, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.258044206084872, 'axref': 'x', 'ay': -0.7494618730467777, 'ayref': 'y', 'showarrow': True, 'x': 0.6126297057157745, 'xref': 'x', 'y': -0.6854975613843457, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': -0.6146493715047701, 'axref': 'x', 'ay': 0.38691359198256975, 'ayref': 'y', 'showarrow': True, 'x': -0.38751610246991464, 'xref': 'x', 'y': 
#0.3370999020682739, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': -1.0144723242694844, 'axref': 'x', 'ay': 0.5606801216598708, 'ayref': 'y', 'showarrow': True, 'x': -0.8627565017528479, 'xref': 'x', 'y': 0.610002960468695, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': -0.8627565017528479, 'axref': 'x', 'ay': 0.610002960468695, 'ayref': 'y', 'showarrow': True, 'x': -0.6146493715047701, 'xref': 'x', 'y': 0.38691359198256975, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': -0.8627565017528479, 'axref': 'x', 'ay': 0.610002960468695, 'ayref': 'y', 'showarrow': True, 'x': -0.5358132139862177, 'xref': 'x', 'y': 0.5379539613624407, 'yref': 
#'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.5543089335815549, 'axref': 'x', 'ay': -0.51238298813099, 'ayref': 'y', 'showarrow': True, 'x': 0.8473938710056763, 'xref': 'x', 'y': -0.41185579759893176, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.6126297057157745, 'axref': 'x', 'ay': -0.6854975613843457, 'ayref': 'y', 'showarrow': True, 'x': 0.5543089335815549, 'xref': 'x', 'y': -0.51238298813099, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.3681749928959967, 'axref': 'x', 'ay': -0.3768978829172026, 'ayref': 'y', 'showarrow': True, 'x': 0.5543089335815549, 'xref': 'x', 'y': -0.51238298813099, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.30285228475452236, 'axref': 'x', 'ay': 0.47079265818833915, 'ayref': 'y', 'showarrow': True, 'x': 0.1637870091026683, 'xref': 'x', 'y': 0.6221171703544651, 'yref': 'y'}, {'arrowhead': 2, 'arrowsize': 1, 'arrowwidth': 1.5, 'ax': 0.1637870091026683, 'axref': 'x', 'ay': 0.6221171703544651, 'ayref': 'y', 'showarrow': True, 'x': -0.16732325338912002, 'xref': 'x', 'y': 0.4350074752713983, 'yref': 'y'}]}}
#pio.show(fig2)