#Der folgende code erhält als Input eine XES-Datei und erstellt mittels PM4PY ein Heuristisches Netz / dfg (zeigt an wie oft eine Message in dem Prozess vorkommt und wie oft eine Message auf eine andere folgt)
#Die Knoten und Kanten werden extrahiert und in das richtige Format gebracht, sodass der Graph mittels React-Flow dargestellt werden kann.
#Der code ist darauf ausgelegt, dass die XES-Datei nur Daten eines Prozesses enthält (Möglich durch den code zur Jobeinteilung)
import plotly.graph_objects as go
import re
import pandas as pd
import pm4py
import plotly.io as pio
from collections import OrderedDict
from google.colab import files
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery

#Upload nur notwendig, wenn code und XES-Datei noch nicht am selben Ort gespeichert sind (bspw. neue Runtime bei google colab)
uploaded = files.upload()

#Der folgende Dateiname muss entsprechend angespasst werden, wenn eine andere XES-Datei analysiert werden soll
log = pm4py.read_xes('seq1and2MachineB-exported.xes')
log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], format='mixed', errors='coerce')
heu_net = pm4py.discover_heuristics_net(log)

dic = heu_net.__dict__
#print(dic)

#Welche Messages kommen in dem Prozess vor und wie oft
messages_nodes_dict = dict(dic.get('activities_occurrences', set()))
#print(messages_nodes_dict)


#Welche Message folgt auf eine andere und wie oft
messages_edges_dict = dict(dic.get('dfg', set()))
#print(messages_edges_dict)



# Wähle relevante Spalten aus, um Masseges mit Mapping verknüpfen zu können
relevant_columns = ['MsgValueDE', 'MsgRank']

# Erstelle einen DataFrame mit den ausgewählten Spalten
msg_rank_df = log[relevant_columns]


# Entferne Duplikate, um eindeutige Paare von MsgValueDE und MsgRank zu erhalten
msg_rank_df = msg_rank_df.drop_duplicates()

#Mapping von K&B, welches Messages auf Basis ihres Ranks Farben zuordnet
mapping = {
    -1: '#BDBDBD',
    1: '#82FA58',
    2: '#b0721e',
    3: '#0B610B',
    4: '#2E64FE',
    5: '#00FFFF',
    6: '#ffffff',
    99: '#A4A4A4',
    101: '#ACFA58',
    102: '#bb7633',
    103: '#3f7633',
    104: '#0000FF',
    105: '#4aeaff',
    106: '#ffffff'
}

msg_rank_df['Color'] = msg_rank_df['MsgRank'].map(mapping)
# Zeige den DataFrame an
#print(msg_rank_df)

#Zu den einzelnen Knoten werden die für die React-App erforderlichen Attribute erstellt und gespeichert
heu_nodes_info = []
for key, value in messages_nodes_dict.items():

    label_node = f"{key}({value})"
    node_id = f"{key}"

    if 'start' in node_id:
        style = {'backgroundColor': '#CAFF70'}
    elif 'end' in node_id:
        style = {'backgroundColor': '#FA5858'}
    else:
        # Hier wird die Farbe aus dem DataFrame basierend auf MsgRank zugewiesen
        cleaned_node_id = node_id.split('(')[0]
        #print(f'Cleaned Node ID: {cleaned_node_id}')
        matching_rows = msg_rank_df['MsgValueDE'].str.contains(cleaned_node_id)
        #print(f'Matching Rows: {matching_rows}')
        color = msg_rank_df.loc[matching_rows, 'Color'].values
        #print(f'Color: {color}')
        style = {'backgroundColor': color[0]} if len(color) > 0 else {}

    node_info = {
        'id': node_id,
        'sourcePosition': 'right',
        'targetPosition': 'left',
        'data': {'label': label_node},
        'style': style
    }
    heu_nodes_info.append(node_info)

#print(heu_nodes_info)


#.js-Datei mit den zuvor erstellten Knoten und jeweiligen Attributen generieren.
#Der "start" und "end" Knoten wird zusätzlich hinzugefügt.
  #Erstellung muss extra erfolgen, da die beiden Knoten bei dieser Analysevarinate von PM4PY nicht automatisch erzeugt werden
output_lines = []
output_lines.append("import { MarkerType } from 'reactflow';\n")
output_lines.append("export const allNodes=[")
output_lines.append(f"{{ id: 'start', sourcePosition: 'right', targetPosition: 'left', data: {{ label: 'start(1)' }}, style: {{'backgroundColor': '#CAFF70'}} }},")
output_lines.append(f"{{ id: 'end', sourcePosition: 'right', targetPosition: 'left', data: {{ label: 'end(1)' }}, style: {{'backgroundColor': '#FA5858'}} }},")
for node_info in heu_nodes_info:
    output_lines.append(f"{{ id: '{node_info['id']}', sourcePosition: '{node_info['sourcePosition']}', targetPosition: '{node_info['targetPosition']}', data: {{ label: '{node_info['data']['label']}' }}, style: {node_info.get('style', {})} }},")
output_lines.append("];")

#Alle Knoten werden in eine JavaScript-Datei geschrieben
with open("data.js", "w") as js_file:
    js_file.write('\n'.join(output_lines))


#Zu den einzelnen Knoten werden die für die React-App erforderlichen Attribute erstellt und gespeichert
flow_edges2 = []

for pair, count in messages_edges_dict.items():

    edge_values = [id for id in f"[{'; '.join(pair)}:{count}"[1:-1].split(';')]
    if len(edge_values) == 2:
        source_id, target_id_with_count = edge_values

        target_id, target_count = target_id_with_count.rsplit(':', 1)

        # Erstelle das Kantenobjekt und füge es zur Liste hinzu
        flow_edge = {
            'id': f'{source_id}-{target_id}',
            'source': source_id,
            'target': target_id.lstrip(),
            'type': 'step',
            'label': str(count),  # Verwende die Zählung nur im Label
        }
        flow_edges2.append(flow_edge)
    else:
        # Wenn die Kante nicht gültig ist, gebe eine Fehlermeldung aus
        print(f"Skipping invalid edge format: {pair}")

# Gib die erstellte Liste von Kanten aus
#print(flow_edges2)

#erste und letzte Aktivität heraussuchen, da diese nicht in dfg enthalten sind
messages_start_str = str(dic.get('start_activities', set()))
messages_start_str = list(eval(messages_start_str))[0].keys()
first_activity = list(messages_start_str)[0]
#print(first_activity)
messages_end_str = str(dic.get('end_activities', set()))
messages_end_str = list(eval(messages_end_str))[0].keys()
last_activity = list(messages_end_str)[0]
#print(last_activity)

#Alle Kanten werden in die selbe .js-Datei wie die Knoten geschrieben
#Zusätzlich werden start und end Knoten mit der ersten bzw. letzten Message verbunden.
  #Dies ist notwendig, da PM4PY diese Kanten in diesem Fall nicht berücksichtigt
output_lines_edges = []
output_lines_edges.append("export const allEdges = [")
output_lines_edges.append(f"{{id: 'startedge', source: 'start', target: '{first_activity}', markerEnd: {{type: MarkerType.ArrowClosed, width: 30, height: 30}}, type: 'step', label: '1'}},")
output_lines_edges.append(f"{{id: 'endedge', source: '{last_activity}', target: 'end', markerEnd: {{type: MarkerType.ArrowClosed, width: 30, height: 30}}, type: 'step', label: '1'}},")
for flow_edge in flow_edges2:
    output_lines_edges.append(f"{{id: '{flow_edge['id']}', source: '{flow_edge['source']}', target: '{flow_edge['target']}', markerEnd: {{type: MarkerType.ArrowClosed, width: 30, height: 30}}, type: '{flow_edge['type']}', label: '{flow_edge['label']}'}},") #animated: true,
output_lines_edges.append("];")
#print(output_lines_edges)

#Die erstellten Kanten werden in die bereits vorhandene .js-Datei geschrieben
with open("data.js", "a") as js_file:
    js_file.write('\n'.join(output_lines_edges))