import React, {useCallback, useLayoutEffect } from 'react';
import ReactFlow, {ReactFlowProvider, MiniMap, Controls, Background, useNodesState, useEdgesState } from 'reactflow';
import ELK from 'elkjs/lib/elk.bundled.js';
import './styles.css';
import 'reactflow/dist/style.css';
import {allNodes, allEdges } from './data.js';

const elkLayout = new ELK();//neue Instanz von ELK-Layout
const elkgraphdesign = {
  //Folgende Layout-Optionen für BPMN-Variante
  
  /*'elk.algorithm': 'layered',
  'nodePlacement.strategy': 'SIMPLE',
  'elk.spacing.nodeNode': '80',
  'elk.direction': 'RIGHT',
  'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
  'elk.layered.crossingMinimization.strategy': 'NONE',
  'elk.layered.spacing.nodeNodeBetweenLayers': '100',
  'elk.spacing.edgeNode': '30',
  'elk.layered.spacing.edgeNodeBetweenLayers': '30',
  'elk.spacing.edgeEdge': '30',
  'elk.layered.crossingMinimization.forceNodeModelOrder': true,*/

  //Folgende Layout-Optionen für Heuristisches Netz
    //keine 

  //Folgende Layout-Optionen für Prozessstränge (Traces) basierend auf CheckIn-Zeit
  'elk.algorithm': 'layered',
  'nodePlacement.strategy': 'SIMPLE',
  'elk.spacing.nodeNode': '100',
  'elk.direction': 'RIGHT',
  'elk.layered.spacing.nodeNodeBetweenLayers': '100',
  'elk.spacing.edgeNode': '30',
  'elk.layered.spacing.edgeNodeBetweenLayers': '30',
  'elk.spacing.edgeEdge': '30',
};

//Position der einzelnen Knoten und Knaten berechenen
const getLayoutedElements = async (graphNodes, graphEdges, graphdesign = {}) => {
  const elkgraph = {
    id: 'root',
    layoutOptions: graphdesign,
    children: graphNodes.map((node) => ({ ...node, width: 150, height: 50 })),
    edges: graphEdges,
  };

  try {
    const elklayoutedGraph = await elkLayout.layout(elkgraph);
    return {
      nodes: elklayoutedGraph.children.map((node) => ({ ...node, position: { x: node.x, y: node.y } })),
      edges: elklayoutedGraph.edges,
    };
  } catch (error) {
    console.error(error);
  }
};

//Verwaltung des Zustandes von Knoten und Kanten durch useNodes/EdgesState
function ProcessLayout() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);//nodes speichert infos über Knoten; setNodes aktualisiert Nodes; ermöglicht Änderungen an Nodes
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  //Layout infos werden durch folgendes aktualisiert
  const handleLayout = useCallback(({ useallNodes = false }) => {
    const options = { ...elkgraphdesign };
    const data = useallNodes ? { nodes: allNodes, edges: allEdges } : { nodes, edges };//vordefinierte oder aktuelle Zustände von Knoten und Kanten

    //Layoutupdate
    getLayoutedElements(data.nodes, data.edges, options).then(({ nodes: layoutedNodes, edges: layoutedEdges }) => {
      setNodes(layoutedNodes);
      setEdges(layoutedEdges);
    });
  }, [nodes, edges]);

  //Layout wird beim ersten rendern initialisiert
  useLayoutEffect(() => {
    handleLayout({ useallNodes: true });
  }, []);

  //return der react-flow Komponente, die das eigentliche Diagramm enthält
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      fitView
      minZoom={0.2}
    >
      <Controls />
      <Background variant="dots" gap={12} size={1.5} />
      <MiniMap />
    </ReactFlow>
  );
}

//Reactflowprovider verwaltet den Zustand des Diagramms
export default () => (
  <ReactFlowProvider>
    <ProcessLayout />
  </ReactFlowProvider>
);
