#Der folgende code erhält als Input eine XES-Datei (die auch Daten mehrerer Prozesse enthalten kann) und gibt erstellt eine .js-Datei,
#die die Trace(s) der in der XES-Datei enthaltenen Prozesse enthält.
#Die .js-Datei wird anschließend wiederverwendet, um den Graph als React-App darzustellen
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

#Trace(s) herauslesen
fp_trace_by_trace = footprints_discovery.apply(log, variant=footprints_discovery.Variants.TRACE_BY_TRACE)
#print(fp_trace_by_trace)

#Speichern der Trace(s) in einer liste.
trace_dash = []
for index, trace_info in enumerate(fp_trace_by_trace):
    trace_for_dash = list(trace_info['trace'])
    trace_dash.append(trace_for_dash)

#print(trace_dash)

# Wähle relevante Spalten aus, um Messages mit der richtigen Farbe aus dem Mapping verknüpfen zu können
relevant_columns = ['MsgValueDE', 'MsgRank']

# Erstelle einen DataFrame mit den ausgewählten Spalten
msg_rank_df = log[relevant_columns]

# Entferne Duplikate, um eindeutige Paare von MsgValueDE und MsgRank zu erhalten
msg_rank_df = msg_rank_df.drop_duplicates()

#Farb-Mapping von K&B
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


#Liste mit evtl. meheren Teillisten, die die verschiedenen Traces enthalten; jedes Element hat noch einen Index (Trace, Position)
traces_list = []
for index, trace_info in enumerate(fp_trace_by_trace):
    trace = list(trace_info['trace'])
    updated_trace = [f'{step}({index},{i})' if i is not None else step for i, step in enumerate(trace)]
    traces_list.append(updated_trace)

#print(traces_list)

#Knoten werden nicht aus dem 'activities' -Teil genommen, da dort der Index fehlt, dieser aber für die eindeutige Identifizierung in der React-Flow-App notwendig ist
nodes_from_traces = []
for sublist in traces_list:
    nodes_from_traces.extend(sublist)

#print(nodes_from_traces)


#start-ziel-paare werden erstellt und in einer Liste gespeichert
source_target_pairs = []

for trace in traces_list:
    pairs = [f'[{trace[i]}; {trace[i+1]}]' for i in range(len(trace)-1)]
    source_target_pairs.extend(pairs)

#print(source_target_pairs)

#Knoten mit ihren jeweiligen Attributen für die React-App werden gespeichert
trace_nodes_info = []
for node_id_parts in nodes_from_traces:#bpmn_nodes:

    if node_id_parts == 'start':
        style = {'backgroundColor': '#CAFF70'}
    elif node_id_parts == 'end':
        style = {'backgroundColor': '#FA5858'}
    else:
        cleaned_node_id = node_id_parts.split('(')[0]
        #print(f'Cleaned Node ID: {cleaned_node_id}')
        matching_rows = msg_rank_df['MsgValueDE'].str.contains(cleaned_node_id)
        #print(f'Matching Rows: {matching_rows}')
        color = msg_rank_df.loc[matching_rows, 'Color'].values
        #print(f'Color: {color}')
        style = {'backgroundColor': color[0]} if len(color) > 0 else {}

    node_info = {
        'id':node_id_parts,
        'sourcePosition': 'right',
        'targetPosition': 'left',
        'data': {'label': node_id_parts},
        'style': style
    }
    trace_nodes_info.append(node_info)
#print(trace_nodes_info)


#Liste mit allen Knoten generieren, die anschließend in die .js-Datei geschrieben werden
#start- und end-Knoten müssen separat erstellt werden
output_lines = []
output_lines.append("import { MarkerType } from 'reactflow';\n")
output_lines.append("export const allNodes=[")
output_lines.append(f"{{ id: 'start', sourcePosition: 'right', targetPosition: 'left', data: {{ label: 'start' }}, style: {{'backgroundColor': '#CAFF70'}} }},")
output_lines.append(f"{{ id: 'end', sourcePosition: 'right', targetPosition: 'left', data: {{ label: 'end' }}, style: {{'backgroundColor': '#FA5858'}} }},")
for node_info in trace_nodes_info:
    output_lines.append(f"{{ id: '{node_info['id']}', sourcePosition: '{node_info['sourcePosition']}', targetPosition: '{node_info['targetPosition']}', data: {{ label: '{node_info['data']['label']}' }}, style: {node_info.get('style', {})} }},")
output_lines.append("];")

#Schreiben der Knoten in eine JavaScript-Datei
with open("data.js", "w") as js_file:
    js_file.write('\n'.join(output_lines))


#Bei der trace_by_trace-variante fehlen die Start- und End-Knoten und somit auch die entsprechenden edges
#diese werden im folgenden erstellt
start_activities_list = [list(sequence['start_activities'])[0] for sequence in fp_trace_by_trace]
#print(start_activities_list)

end_activities_list = [list(sequence['end_activities'])[0] for sequence in fp_trace_by_trace]
#print(end_activities_list)

min_trace_lengths = [entry['min_trace_length'] for entry in fp_trace_by_trace]
#print(min_trace_lengths)


#Kanten zwischen den einzelnen Knoten mit den jeweiligen Attributen für React-Flow erstellen und speichern
trace_flow_edges = []
for flow_id, flow_edge_str in enumerate(source_target_pairs):
    edge_values = [id for id in flow_edge_str[1:-1].split(';')]
    if len(edge_values) == 2:
        source_id, target_id = edge_values
        flow_edge = {
          'id': f'{source_id}-{target_id}',
          'source': source_id,
          'target': target_id.lstrip(),
          #'animated': True,
          'type': 'step'
        }
        trace_flow_edges.append(flow_edge)
    else:
        print(f"Skipping invalid edge format: {flow_edge_str}")

#Alle Kanten, die in die .js-Datei geschrieben werden sollen, werden gespeichert
#Kanten vom Start- und End-Knoten zu den jeweils ersten bzw. letzten Messages jeder Trace werden erstellt
output_lines_edges = []
output_lines_edges.append("export const allEdges = [")
for index, flow_start_edge in enumerate(start_activities_list):
    output_lines_edges.append(f"{{id: '{flow_start_edge}{index}', source: 'start', target: '{flow_start_edge}({index},0)', markerEnd: {{type: MarkerType.ArrowClosed, width: 30, height: 30}}, type: 'step'}},")
for flow_edge in trace_flow_edges:
    output_lines_edges.append(f"{{id: '{flow_edge['id']}', source: '{flow_edge['source']}', target: '{flow_edge['target']}', markerEnd: {{type: MarkerType.ArrowClosed, width: 30, height: 30}}, type: '{flow_edge['type']}'}},")
for index, flow_end_edge in enumerate(end_activities_list):
    output_lines_edges.append(f"{{id: '{flow_end_edge}{index}', source: '{flow_end_edge}({index},{min_trace_lengths[index]-1})', target: 'end', markerEnd: {{type: MarkerType.ArrowClosed, width: 30, height: 30}}, type: 'step'}},")
output_lines_edges.append("];")

#Hinzufügen der Kanten in die bereits bestehende .js-Datei
with open("data.js", "a") as js_file:
    js_file.write('\n'.join(output_lines_edges))