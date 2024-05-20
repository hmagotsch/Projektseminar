//BPMN-Modell
import { MarkerType } from 'reactflow';

export const allNodes=[
{ id: 'start', sourcePosition: 'right', targetPosition: 'left', data: { label: 'start' }, style: {'backgroundColor': '#CAFF70'} },
{ id: 'xor_1_split', sourcePosition: 'right', targetPosition: 'left', data: { label: 'xor_1_split' }, style: {} },
{ id: 'Initiator Vorposition', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Initiator Vorposition' }, style: {'backgroundColor': '#ffffff'} },
{ id: 'Rolloeinfahrt auf Vorposition', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Rolloeinfahrt auf Vorposition' }, style: {'backgroundColor': '#ffffff'} },
{ id: 'xor_1_join', sourcePosition: 'right', targetPosition: 'left', data: { label: 'xor_1_join' }, style: {} },
{ id: 'xor_2_join', sourcePosition: 'right', targetPosition: 'left', data: { label: 'xor_2_join' }, style: {} },
{ id: 'Zähler und Maschinengeschwindigkeit', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Zähler und Maschinen- geschwindigkeit' }, style: {'backgroundColor': '#A4A4A4'} },
{ id: 'xor_2_split', sourcePosition: 'right', targetPosition: 'left', data: { label: 'xor_2_split' }, style: {} },
{ id: 'Stapel frei bei Nonstop', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Stapel frei bei Nonstop' }, style: {'backgroundColor': '#A4A4A4'} },
{ id: 'Schutz Feuchtwerk geöffnet', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Schutz Feuchtwerk geöffnet' }, style: {'backgroundColor': '#2E64FE'} },
{ id: 'xor_3_split', sourcePosition: 'right', targetPosition: 'left', data: { label: 'xor_3_split' }, style: {} },
{ id: 'Aufforderung Überschießsperre quittieren', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Aufforderung Überschießsperre quittieren' }, style: {'backgroundColor': '#A4A4A4'} },
{ id: 'Bogen zu weit Seite 1', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Bogen zu weit Seite 1' }, style: {'backgroundColor': '#bb7633'} },
{ id: 'Schnellhalt Maschine', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Schnellhalt Maschine' }, style: {'backgroundColor': '#0000FF'} },
{ id: 'Reserve j21200_w27_b03', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Reserve j21200_w27_b03' }, style: {'backgroundColor': '#ffffff'} },
{ id: 'Nachlauf Fortdruck', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Nachlauf Fortdruck' }, style: {'backgroundColor': '#A4A4A4'} },
{ id: 'xor_3_join', sourcePosition: 'right', targetPosition: 'left', data: { label: 'xor_3_join' }, style: {} },
{ id: 'end', sourcePosition: 'right', targetPosition: 'left', data: { label: 'end' }, style: {'backgroundColor': '#FA5858'} },
];export const allEdges = [
{id: 'xor_1_join-xor_2_join', source: 'xor_1_join', target: 'xor_2_join', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'xor_2_split-xor_3_split', source: 'xor_2_split', target: 'xor_3_split', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'xor_1_split-Rolloeinfahrt auf Vorposition', source: 'xor_1_split', target: 'Rolloeinfahrt auf Vorposition', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Rolloeinfahrt auf Vorposition-xor_1_join', source: 'Rolloeinfahrt auf Vorposition', target: 'xor_1_join', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Zähler und Maschinengeschwindigkeit-xor_2_split', source: 'Zähler und Maschinengeschwindigkeit', target: 'xor_2_split', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'start-xor_1_split', source: 'start', target: 'xor_1_split', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Schnellhalt Maschine-Reserve j21200_w27_b03', source: 'Schnellhalt Maschine', target: 'Reserve j21200_w27_b03', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'xor_1_split-Initiator Vorposition', source: 'xor_1_split', target: 'Initiator Vorposition', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Nachlauf Fortdruck-xor_3_join', source: 'Nachlauf Fortdruck', target: 'xor_3_join', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'xor_3_split-Bogen zu weit Seite 1', source: 'xor_3_split', target: 'Bogen zu weit Seite 1', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'xor_3_join-end', source: 'xor_3_join', target: 'end', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Aufforderung Überschießsperre quittieren-xor_3_join', source: 'Aufforderung Überschießsperre quittieren', target: 'xor_3_join', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Schutz Feuchtwerk geöffnet-xor_2_join', source: 'Schutz Feuchtwerk geöffnet', target: 'xor_2_join', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'xor_2_split-Stapel frei bei Nonstop', source: 'xor_2_split', target: 'Stapel frei bei Nonstop', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Reserve j21200_w27_b03-Nachlauf Fortdruck', source: 'Reserve j21200_w27_b03', target: 'Nachlauf Fortdruck', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'xor_3_split-Aufforderung Überschießsperre quittieren', source: 'xor_3_split', target: 'Aufforderung Überschießsperre quittieren', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Stapel frei bei Nonstop-Schutz Feuchtwerk geöffnet', source: 'Stapel frei bei Nonstop', target: 'Schutz Feuchtwerk geöffnet', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Bogen zu weit Seite 1-Schnellhalt Maschine', source: 'Bogen zu weit Seite 1', target: 'Schnellhalt Maschine', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Initiator Vorposition-xor_1_join', source: 'Initiator Vorposition', target: 'xor_1_join', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'xor_2_join-Zähler und Maschinengeschwindigkeit', source: 'xor_2_join', target: 'Zähler und Maschinengeschwindigkeit', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
];


//Directly-Follows-Graph
/*import { MarkerType } from 'reactflow';

export const allNodes=[
{ id: 'start', sourcePosition: 'right', targetPosition: 'left', data: { label: 'start(1)' }, style: {'backgroundColor': '#CAFF70'} },
{ id: 'end', sourcePosition: 'right', targetPosition: 'left', data: { label: 'end(1)' }, style: {'backgroundColor': '#FA5858'} },
{ id: 'Zähler und Maschinengeschwindigkeit', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Zähler und Maschinen- geschwindigkeit(2)' }, style: {'backgroundColor': '#A4A4A4'} },
{ id: 'Rolloeinfahrt auf Vorposition', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Rolloeinfahrt auf Vorposition(1)' }, style: {'backgroundColor': '#ffffff'} },
{ id: 'Stapel frei bei Nonstop', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Stapel frei bei Nonstop(1)' }, style: {'backgroundColor': '#A4A4A4'} },
{ id: 'Schutz Feuchtwerk geöffnet', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Schutz Feuchtwerk geöffnet(1)' }, style: {'backgroundColor': '#2E64FE'} },
{ id: 'Aufforderung Überschießsperre quittieren', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Aufforderung Überschießsperre quittieren(1)' }, style: {'backgroundColor': '#A4A4A4'} },
//{ id: 'Initiator Vorposition', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Initiator Vorposition(1)' }, style: {'backgroundColor': '#ffffff'} },
//{ id: 'Bogen zu weit Seite 1', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Bogen zu weit Seite 1(1)' }, style: {'backgroundColor': '#bb7633'} },
//{ id: 'Schnellhalt Maschine', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Schnellhalt Maschine(1)' }, style: {'backgroundColor': '#0000FF'} },
//{ id: 'Reserve j21200_w27_b03', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Reserve j21200_w27_b03(1)' }, style: {'backgroundColor': '#ffffff'} },
//{ id: 'Nachlauf Fortdruck', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Nachlauf Fortdruck(1)' }, style: {'backgroundColor': '#A4A4A4'} },
];export const allEdges = [
{id: 'startedge', source: 'start', target: 'Rolloeinfahrt auf Vorposition', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
{id: 'endedge', source: 'Aufforderung Überschießsperre quittieren', target: 'end', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
//{id: 'Bogen zu weit Seite 1- Schnellhalt Maschine', source: 'Bogen zu weit Seite 1', target: 'Schnellhalt Maschine', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
//{id: 'Initiator Vorposition- Zähler und Maschinengeschwindigkeit', source: 'Initiator Vorposition', target: 'Zähler und Maschinengeschwindigkeit', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
//{id: 'Reserve j21200_w27_b03- Nachlauf Fortdruck', source: 'Reserve j21200_w27_b03', target: 'Nachlauf Fortdruck', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
{id: 'Rolloeinfahrt auf Vorposition- Zähler und Maschinengeschwindigkeit', source: 'Rolloeinfahrt auf Vorposition', target: 'Zähler und Maschinengeschwindigkeit', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
//{id: 'Schnellhalt Maschine- Reserve j21200_w27_b03', source: 'Schnellhalt Maschine', target: 'Reserve j21200_w27_b03', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
{id: 'Schutz Feuchtwerk geöffnet- Zähler und Maschinengeschwindigkeit', source: 'Schutz Feuchtwerk geöffnet', target: 'Zähler und Maschinengeschwindigkeit', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
{id: 'Stapel frei bei Nonstop- Schutz Feuchtwerk geöffnet', source: 'Stapel frei bei Nonstop', target: 'Schutz Feuchtwerk geöffnet', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
{id: 'Zähler und Maschinengeschwindigkeit- Aufforderung Überschießsperre quittieren', source: 'Zähler und Maschinengeschwindigkeit', target: 'Aufforderung Überschießsperre quittieren', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
//{id: 'Zähler und Maschinengeschwindigkeit- Bogen zu weit Seite 1', source: 'Zähler und Maschinengeschwindigkeit', target: 'Bogen zu weit Seite 1', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
{id: 'Zähler und Maschinengeschwindigkeit- Stapel frei bei Nonstop', source: 'Zähler und Maschinengeschwindigkeit', target: 'Stapel frei bei Nonstop', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step', label: '1'},
];*/

//Traces
/*import { MarkerType } from 'reactflow';

export const allNodes=[
{ id: 'start', sourcePosition: 'right', targetPosition: 'left', data: { label: 'start' }, style: {'backgroundColor': '#CAFF70'} },
{ id: 'end', sourcePosition: 'right', targetPosition: 'left', data: { label: 'end' }, style: {'backgroundColor': '#FA5858'} },
{ id: 'Rolloeinfahrt auf Vorposition(0,0)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Rolloeinfahrt auf Vorposition(1,0)' }, style: {'backgroundColor': '#ffffff'} },
{ id: 'Zähler und Maschinengeschwindigkeit(0,1)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Zähler und Maschinen- geschwindigkeit(1,1)' }, style: {'backgroundColor': '#A4A4A4'} },
{ id: 'Stapel frei bei Nonstop(0,2)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Stapel frei bei Nonstop(1,2)' }, style: {'backgroundColor': '#A4A4A4'} },
{ id: 'Schutz Feuchtwerk geöffnet(0,3)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Schutz Feuchtwerk geöffnet(1,3)' }, style: {'backgroundColor': '#2E64FE'} },
{ id: 'Zähler und Maschinengeschwindigkeit(0,4)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Zähler und Maschinen- geschwindigkeit(1,4)' }, style: {'backgroundColor': '#A4A4A4'} },
{ id: 'Aufforderung Überschießsperre quittieren(0,5)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Aufforderung Überschießsperre quittieren(1,5)' }, style: {'backgroundColor': '#A4A4A4'} },
{ id: 'Initiator Vorposition(1,0)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Initiator Vorposition(2,0)' }, style: {'backgroundColor': '#ffffff'} },
{ id: 'Zähler und Maschinengeschwindigkeit(1,1)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Zähler und Maschinen- geschwindigkeit(2,1)' }, style: {'backgroundColor': '#A4A4A4'} },
{ id: 'Bogen zu weit Seite 1(1,2)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Bogen zu weit Seite 1(2,2)' }, style: {'backgroundColor': '#bb7633'} },
{ id: 'Schnellhalt Maschine(1,3)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Schnellhalt Maschine(2,3)' }, style: {'backgroundColor': '#0000FF'} },
{ id: 'Reserve j21200_w27_b03(1,4)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Reserve j21200_w27_b03(2,4)' }, style: {'backgroundColor': '#ffffff'} },
{ id: 'Nachlauf Fortdruck(1,5)', sourcePosition: 'right', targetPosition: 'left', data: { label: 'Nachlauf Fortdruck(2,5)' }, style: {'backgroundColor': '#A4A4A4'} },
];export const allEdges = [
{id: 'Rolloeinfahrt auf Vorposition0', source: 'start', target: 'Rolloeinfahrt auf Vorposition(0,0)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Initiator Vorposition1', source: 'start', target: 'Initiator Vorposition(1,0)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Rolloeinfahrt auf Vorposition(0,0)- Zähler und Maschinengeschwindigkeit(0,1)', source: 'Rolloeinfahrt auf Vorposition(0,0)', target: 'Zähler und Maschinengeschwindigkeit(0,1)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Zähler und Maschinengeschwindigkeit(0,1)- Stapel frei bei Nonstop(0,2)', source: 'Zähler und Maschinengeschwindigkeit(0,1)', target: 'Stapel frei bei Nonstop(0,2)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Stapel frei bei Nonstop(0,2)- Schutz Feuchtwerk geöffnet(0,3)', source: 'Stapel frei bei Nonstop(0,2)', target: 'Schutz Feuchtwerk geöffnet(0,3)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Schutz Feuchtwerk geöffnet(0,3)- Zähler und Maschinengeschwindigkeit(0,4)', source: 'Schutz Feuchtwerk geöffnet(0,3)', target: 'Zähler und Maschinengeschwindigkeit(0,4)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Zähler und Maschinengeschwindigkeit(0,4)- Aufforderung Überschießsperre quittieren(0,5)', source: 'Zähler und Maschinengeschwindigkeit(0,4)', target: 'Aufforderung Überschießsperre quittieren(0,5)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Initiator Vorposition(1,0)- Zähler und Maschinengeschwindigkeit(1,1)', source: 'Initiator Vorposition(1,0)', target: 'Zähler und Maschinengeschwindigkeit(1,1)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Zähler und Maschinengeschwindigkeit(1,1)- Bogen zu weit Seite 1(1,2)', source: 'Zähler und Maschinengeschwindigkeit(1,1)', target: 'Bogen zu weit Seite 1(1,2)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Bogen zu weit Seite 1(1,2)- Schnellhalt Maschine(1,3)', source: 'Bogen zu weit Seite 1(1,2)', target: 'Schnellhalt Maschine(1,3)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Schnellhalt Maschine(1,3)- Reserve j21200_w27_b03(1,4)', source: 'Schnellhalt Maschine(1,3)', target: 'Reserve j21200_w27_b03(1,4)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Reserve j21200_w27_b03(1,4)- Nachlauf Fortdruck(1,5)', source: 'Reserve j21200_w27_b03(1,4)', target: 'Nachlauf Fortdruck(1,5)', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Aufforderung Überschießsperre quittieren0', source: 'Aufforderung Überschießsperre quittieren(0,5)', target: 'end', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
{id: 'Nachlauf Fortdruck1', source: 'Nachlauf Fortdruck(1,5)', target: 'end', markerEnd: {type: MarkerType.ArrowClosed, width: 30, height: 30}, type: 'step'},
];*/

