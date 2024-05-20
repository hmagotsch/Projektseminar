Folgendes bezieht sich auf die Dateien im Ordner Prozessvisualisierung

+++Der letzte Test und Ausführung der Dateien war der 23.04.2024 kurz vor der Codeübergabe an K&B (25.04.2024)+++
+++Da die Dateien mit einer Vielzahl an Packages arbeiten, kann eine neue Version der Packages Änderungen am Quellcode erfordern+++


Die Dateien bpmn.py, dfg.py und traces.py analysieren jeweils eine hochgeladene XES-Datei (XES-Dateien können durch create_xes_file.py für beliebige 
Druckaufträge erstellt werden) und erstellen eine .js output-Datei, die die einzelnen 
Knoten und Kanten der jeweiligen Darstellungsform enthält. Die Output-Datei wird an dem selben Ort gespeichert, wie die .py-Datei, die ausgeführt wurde.
Die React-Anwendung (App.js, index.js, styles.css) nutzt die Output-Datei als Input und erstellt daraus eine Graphische Darstellung des Prozesses. Wichtig ist,
dass der Name der Output-Datei mit der import-Anweisung in App.js übereinstimmt.


bpmn.py:

#Ziel des folgenden code ist es, die Knoten und Kanten eines mit PM4PY erstellten BPMN-Modells zu extrahieren und als Java-Script-Code in eine .js-Datei zu 
schreiben, welche diese Daten als Input zur Erstellung eines React-Flow-Graphen nutzt
#Der code bekommt als Input eine ausgewählte XES-Datei, welche anschließend mittels PM4PY-Funktionen analysiert wird.
#Es wird ein BPMN_Modell zu der XES-Datei erstellt und die einzelnen Knoten und Kanten werden über das dict-Attribut ausgelesen.
#Die Knoten und Kanten werden durch die Funktion "erstelle_reihenfolge" in eine grobe Ordnung gebracht (Knoten, die in dem Prozess früher vorkommen stehen 
weiter vorne in der Liste), da diese Reihenfolge einen Einfluss auf das Layout des React-Flow-Graphen hat.
#Im nächsten Schritt wird das Farb-Mapping von K&B integriert, welches den einzelnen Messages auf Basis ihres Ranks eine Farbe zuweist
#Am Ende werden die Knoten und Kanten mit ihren für die React-App notwendigen Attributen verknüpft und in eine .js-Datei geschrieben


dfg.py:

#Der folgende code erhält als Input eine XES-Datei und erstellt mittels PM4PY ein Heuristisches Netz / dfg (zeigt an wie oft ein Knoten in dem Prozess 
vorkommt und wie oft ein Knoten auf einen anderen folgt)
#Die Knoten und Kanten werden extrahiert und in das richtige Format gebracht, sodass der Graph mittels React-Flow dargestellt werden kann
#Der code ist darauf ausgelegt, dass die XES-Datei nur Daten eines Prozesses enthält (Möglich durch den code zur Jobeinteilung)


traces.py:

#Der folgende code erhält als Input eine XES-Datei (die auch Daten mehrerer Prozesse enthalten kann) und gibt erstellt eine .js-Datei,
#die die Trace(s) der in der XES-Datei enthaltenen Prozesse enthält.
#Die .js-Datei wird anschließend wieder verwendet, um den Graph als React-App darzustellen


React-App (App.js, index.js, styles.css):

Die React-App stellt den Prozess mittels React-Flow dar. Voraussetzung dafür ist die Installation von Node.js und dem Package Manager npm (Alternativ
ginge auch yarn oder pnpm). Außerdem ist es notwendig react-flow-renderer, elkjs und web-worker zu installieren (über npm). Die React-App kann
anschließend über "npm start" gestartet werden.
In der Datei App.js muss zusätzlich noch darauf geachtet werden, welche Layout-Optionen für die jeweilige Darstellung auskommentiert sind und welche nicht



plotlyNetzwerkgraph.py:

#Es wird aus einer XES-Datei ein Plotly-Netzwerkgraph erstellt
#Idee wurde aber verworfen, da Netzwerkgraphen nicht so gut geeignet sind, um Prozesse darzustellen


seq1and2MachineB-exported.xes:

Beispiel XES.Datei, die momentan in allen drei Dateien (bpmn.py, dfg.py und traces.py) angegeben ist


data.js

Beispieldatei, die als Input für die React-App genutzt werden kann. Die Datei enthält den Output aus bpmn.py, dfg.py und traces.py,
die jeweils mit seq1and2MachineB-exported.xes gearbeitet haben.