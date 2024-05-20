#Ziel des folgenden codes ist es, die Knoten und Kanten eines mit PM4PY erstellten BPMN-Modells zu extrahieren und als Java-Script-Code in eine .js-Datei zu schreiben, welche diese Daten als Input zur Erstellung eines React-Flow-Graphen nutzt
#Der code bekommt als Input eine ausgewählte XES-Datei, welche anschließend mittels PM4PY-Funktionen analysiert wird.
#Es wird ein BPMN-Modell zu der XES-Datei erstellt und die einzelnen Knoten und Kanten werden über das dict-Attribut ausgelesen.
#Die Knoten und Kanten werden durch die Funktion "erstelle_reihenfolge" in eine grobe Ordnung gebracht (Knoten, die in dem Prozess früher vorkommen stehen weiter vorne in der Liste), da diese Reihenfolge einen Einfluss auf das Layout des React-Flow-Graphen hat.
#Im nächsten Schritt wird das Farb-Mapping von K&B integriert, welches den einzelnen Messages auf Basis ihres Ranks eine Farbe zuweist
#Am Ende werden die Knoten und Kanten mit ihren für die React-App notwendigen Attributen verknüpft und in eine .js-Datei geschrieben
import plotly.graph_objects as go
import re
import pandas as pd
import pm4py
import plotly.io as pio
from collections import OrderedDict
from google.colab import files
import json
from collections import defaultdict

#Upload nur notwendig, wenn code und XES-Datei noch nicht am selben Ort gespeichert sind (bspw. neue Runtime bei google colab)
uploaded = files.upload()

#Der folgende Dateiname muss entsprechend angespasst werden, wenn eine andere XES-Datei analysiert werden soll
log=pm4py.read_xes('seq1and2MachineB-exported.xes')
log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], format='mixed', errors='coerce')
process_tree = pm4py.discover_process_tree_inductive(log)
bpmn_model = pm4py.convert_to_bpmn(process_tree)


#dict Attribut des bpmn Modells auslesen (nodes und edges)
dic = bpmn_model.__dict__


#Knoten extrahieren, aufbereiten und in einer liste speichern
bpmn_nodes_str = str(dic.get('_BPMN__nodes', set()))
bpmn_nodes_str = bpmn_nodes_str.replace('}', '')
bpmn_nodes_str = bpmn_nodes_str.replace('{', '')
bpmn_nodes_str = bpmn_nodes_str.replace(', id', '; id')
bpmn_nodes = [node_id.split('@')[-1] for node_id in bpmn_nodes_str.split(';')]
#print(bpmn_nodes)
#print(len(bpmn_nodes))

#Verbindungen extrahieren, aufbereiten und in einer liste speichern
bpmn_flows_str = str(dic.get('_BPMN__flows', set()))
bpmn_flows_str = bpmn_flows_str.replace('}', '')
bpmn_flows_str = bpmn_flows_str.replace('{', '')
bpmn_flows_str = bpmn_flows_str.replace(', id', '; id')
flow_data = [flow.split(' -> ') for flow in bpmn_flows_str.split(';')]
flow_edges = ['[' + '; '.join([id.split('@')[-1] for id in sublist]) + ']' for sublist in flow_data]

source_target_paare=flow_edges

#Reihenfolge der Knoten hat einen Einfluss auf das spätere Layout in React-Flow. Daher wird hier eine grobe Vorsortierung durchgeführt
def erstelle_reihenfolge(source_target_paare):
    graph = defaultdict(list)

    # Erstelle einen gerichteten Graphen aus den Paaren
    for paar in source_target_paare:
        source, target = [aktivitaet.strip(" []") for aktivitaet in paar.split(";")]
        graph[source].append(target)

    def dfs(activity, visited, result):
        visited.add(activity)
        for next_activity in graph.get(activity, []):
            if next_activity not in visited:
                dfs(next_activity, visited, result)
        result.append(activity)

    # Durchführung einer Tiefensuche (DFS) für die Reihenfolge
    start = "start"
    end = "end"
    visited = set()
    reihenfolge = []

    dfs(start, visited, reihenfolge)
    reihenfolge.reverse()  # Da die DFS-Ausgabe umgekehrt ist

    # Überprüfe, ob 'end' bereits in der Reihenfolge ist, bevor es hinzugefügt wird
    if end not in reihenfolge:
        reihenfolge.append(end)

    return reihenfolge

reihenfolge = erstelle_reihenfolge(source_target_paare)


# Wähle relevante Spalten aus, um Messages auf Basis ihres Ranks mit dem Mapping verknüpfen zu können
relevant_columns = ['MsgValueDE', 'MsgRank']

# Erstelle einen DataFrame mit den ausgewählten Spalten
msg_rank_df = log[relevant_columns]

# Entferne Duplikate, um eindeutige Paare von MsgValueDE und MsgRank zu erhalten
msg_rank_df = msg_rank_df.drop_duplicates()

#Mapping von MessageRank zu Farben
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

#zu jedem Knoten werden die nötigen Attribute für die React-Flow-Darstellung ermittelt und gespeichert
bpmn_nodes_info = []
for node_id_parts in reihenfolge:

    if node_id_parts == 'start':
        style = {'backgroundColor': '#CAFF70'}
    elif node_id_parts == 'end':
        style = {'backgroundColor': '#FA5858'}
    else:
        color = msg_rank_df.loc[msg_rank_df['MsgValueDE'] == node_id_parts, 'Color'].values
        style = {'backgroundColor': color[0]} if len(color) > 0 else {}

    node_info = {
        'id':node_id_parts,
        'sourcePosition': 'right',
        'targetPosition': 'left',
        'data': {'label': node_id_parts},
        'style': style
    }
    bpmn_nodes_info.append(node_info)

#print(bpmn_nodes)
#print(bpmn_nodes_info)
#print(len(bpmn_nodes_info))


#Ausgabe generieren
#Die einzelnen Knoten werden im richtigen Format und mit den zuvor ermittelten Attributen in die .js-Datei geschrieben
output_lines = []
output_lines.append("import { MarkerType } from 'reactflow';\n")
output_lines.append("export const allNodes=[")
for node_info in bpmn_nodes_info:
    output_lines.append(f"{{ id: '{node_info['id']}', sourcePosition: '{node_info['sourcePosition']}', targetPosition: '{node_info['targetPosition']}', data: {{ label: '{node_info['data']['label']}' }}, style: {node_info.get('style', {})} }},")
output_lines.append("];")

#Hinzufügen zu einer JavaScript-Datei
with open("data.js", "w") as js_file:
    js_file.write('\n'.join(output_lines))


#zu jeder Kante werden die nötigen Attribute für die React-Flow-Darstellung ermittelt und gespeichert
flow_edges2 = []
for flow_id, flow_edge_str in enumerate(flow_edges):
    edge_values = [id.strip() for id in flow_edge_str[1:-1].split(';')]
    if len(edge_values) == 2:
        source_id, target_id = edge_values

        flow_edge = {
          'id': f'{source_id}-{target_id}',
          'source': source_id,
          'target': target_id,
          'type': 'step'
        }
        flow_edges2.append(flow_edge)
    else:
        print(f"Skipping invalid edge format: {flow_edge_str}")

#Die einzelnen Kanten werden im richtigen Format und mit den zuvor ermittelten Attributen in die .js-Datei geschrieben
output_lines_edges = []
output_lines_edges.append("export const allEdges = [")
for flow_edge in flow_edges2:
    output_lines_edges.append(f"{{id: '{flow_edge['id']}', source: '{flow_edge['source']}', target: '{flow_edge['target']}', markerEnd: {{type: MarkerType.ArrowClosed, width: 30, height: 30}}, type: '{flow_edge['type']}'}},") #animated: true,
output_lines_edges.append("];")

#Die Kanten-Infos werden zu der bereits bestehenden Datei und hinzugefügt, sodass Konten und Kanten in der selben Datei enthalten sind
with open("data.js", "a") as js_file:
    js_file.write('\n'.join(output_lines_edges))