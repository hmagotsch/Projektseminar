import streamlit as st
import pm4py
import pandas as pd
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from collections import Counter
from io import BytesIO
from io import StringIO
import os
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Funktionen für verschiedene Seiten
def home():
    st.title("K&B Analytics")
    st.divider()
    #st.header("Hier sind einige KIP-Kacheln auf der Startseite.")

    # Kacheln erstellen
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("KIP 1")
        st.metric(label="Temperature", value="70 °F", delta="1.2 °F")

    with col2:
        st.subheader("KIP 2")
        st.metric(label="Gas price", value=4, delta=-0.5)

    with col3:
        st.subheader("KIP 3")
        st.metric(label="Active developers", value=123, delta=123)

def page1():
    st.title("KPI")
    # Hier Inhalt für Seite 2 definieren

def page2():
    st.title("Process View")
    
    uploaded_file = st.file_uploader("Upload a XES file", type=["xes"])

    if uploaded_file is not None:
        st.subheader("Uploaded File Content:")
        st.write("\n")
        st.write("\n")

        log = pm4py.read_xes('MachineA_JobNr2_ProductionTime-exported.xes')
        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], format='mixed', errors='coerce')

        # Fußabdrücke für jede Trace berechnen
        fp_trace_by_trace = footprints_discovery.apply(log, variant=footprints_discovery.Variants.TRACE_BY_TRACE)

        # Extrahiere die relevanten Spalten aus dem Log
        relevant_columns = ['MsgValueDE', 'MsgRank']

        # Erstelle einen DataFrame mit den ausgewählten Spalten
        msg_rank_df = log[relevant_columns]

        # Entferne Duplikate, um eindeutige Paare von MsgValueDE und MsgRank zu erhalten
        msg_rank_df = msg_rank_df.drop_duplicates()

        # Mapping für die Farben
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

        # Füge eine Spalte für die Farbe basierend auf dem MSGRank hinzu
        msg_rank_df['Color'] = msg_rank_df['MsgRank'].map(mapping)

        # Zeige den DataFrame an
        #st.write(msg_rank_df)

        # Speichere die Farben, den MsgRank und die Anzahl für jede Trace in trace_dash
        trace_info_list = []

        for trace in fp_trace_by_trace:
            trace_info = pd.DataFrame(trace['trace'], columns=['MsgValueDE'])
            trace_info = pd.merge(trace_info, msg_rank_df, on='MsgValueDE')
            trace_info['MsgValueDECount'] = trace_info.groupby('MsgValueDE')['MsgValueDE'].transform('count')
            trace_info = trace_info[['MsgValueDE', 'MsgRank', 'Color', 'MsgValueDECount']].drop_duplicates()
            trace_info = trace_info.sort_values(by='MsgValueDECount', ascending=False)  # Sortiere nach MsgCount absteigend
            trace_info_list.append(trace_info)

        # Zeige die Informationen für jede Trace in trace_dash an
        #for index, trace_info in enumerate(trace_info_list):
         #   st.write(f"Trace {index + 1}:\n{trace_info}\n")


        trace_dash2 = []
        for index, trace_info in enumerate(fp_trace_by_trace):
            trace_for_dash2 = list(trace_info['trace'])
            trace_dash2.append(trace_for_dash2)

        #st.write(trace_dash2)

        anzahl_traces = len(trace_dash2)

        # Anzahl der Einträge in den inneren Listen (Anzahl der Schritte pro Trace)
        anzahl_schritte_pro_trace = [len(inner_list) for inner_list in trace_dash2]

        gesamtanzahl_schritte = sum(anzahl_schritte_pro_trace)

        
        col1, col2, col3 = st.columns(3)
        col1.metric("Anzahl der Traces", anzahl_traces)
        col2.metric("Gesamtzahl aller Messages", gesamtanzahl_schritte)
        col3.metric("Anzahl der Messages pro Trace", str(anzahl_schritte_pro_trace))
        

        global_start_time = log['time:timestamp'].min()
        global_end_time = log['time:timestamp'].max()

        st.write("\n")
        st.write("\n")

        col4, col5 = st.columns(2)
        col4.metric("Startdate", global_start_time.strftime('%Y-%m-%d'))
        col5.metric("Starttime", global_start_time.strftime('%H:%M:%S.%f'))

        st.write("\n")
        st.write("\n")

        col6, col7 = st.columns(2)
        col6.metric("Enddate:", global_end_time.strftime('%Y-%m-%d'))
        col7.metric("Endtime:", global_end_time.strftime('%H:%M:%S.%f'))

        st.write("\n")
        st.write("\n")



        # Erstelle eine leere Liste, um die Top 10 MsgValueDE mit Count zu speichern
        top_10_msgs_list = []

        # Durchlaufe jede Trace-Information in trace_info_list
        for trace_info in trace_info_list:
            # Extrahiere die Top 10 MsgValueDE mit den höchsten Count-Werten
            top_10_msgs = trace_info.head(10)[['MsgValueDE', 'MsgValueDECount']]

            # Füge die extrahierten Informationen zur Liste hinzu
            top_10_msgs_list.append(top_10_msgs)

        # Zeige die Liste mit den Top 10 MsgValueDE für jede Trace an
        for index, top_10_msgs in enumerate(top_10_msgs_list):
            st.write(f"Top 10 MsgValueDE für Trace {index + 1}:")
            st.table(top_10_msgs)


        trace_msg_rank_count = []

        # Iteriere durch jede Trace-Information in trace_info_list
        for trace_info in trace_info_list:
            # Gruppiere nach 'MsgRank' und summiere 'MsgValueDECount' für jede Gruppe
            msg_rank_count = trace_info.groupby('MsgRank')['MsgValueDECount'].sum().reset_index()

            # Füge die Counter-Informationen zur Liste hinzu
            trace_msg_rank_count.append(msg_rank_count)

        # Erstellen Sie die DataFrame mit den gewünschten Einträgen
        df = pd.DataFrame({
            "Message Rank": ["< 99", -1, 1, 2, 3, 4, 5, 6, "> 98", 99, 101, 102, 103, 104, 105, 106],
            "Farbe": ["-", "#BDBDBD", "#82FA58", "#b0721e", "#0B610B", "#2E64FE", "#00FFFF", "#ffffff", "-", "#A4A4A4", "#ACFA58", "#bb7633", "#3f7633", "#0000FF", "#4aeaff", "#ffffff"],
            "Bedeutung": ["durch den Drucker behebbare Meldungen", "noch nicht zugeordnet", "Information", "Warnung", "eingeschränkte Produktion möglich", "keine Produktion möglich", "nicht zuordenbar", "Info nur im Logfile", "durch den Drucker nicht behebare Meldungen", "noch nicht zugeordnet", "Information", "Warnung", "eingeschränkte Produktion möglich", "keine Produktion möglich", "nicht zuordenbar", "Info nur im Logfile"],
        })

        # Iteriere durch jede Trace-Information in trace_msg_rank_count und fülle den DataFrame
        for index, msg_rank_count_df in enumerate(trace_msg_rank_count):
            msg_rank_count_df = msg_rank_count_df.rename(columns={'MsgValueDECount': f'Trace {index + 1}'})
            df = pd.merge(df, msg_rank_count_df, how='left', left_on='Message Rank', right_on='MsgRank')
            df = df.drop(columns=['MsgRank'])

        # Fülle NaN-Werte mit 0
        df = df.fillna(0)

        column_widths = {
        "A": 50,  
        "B": 350, 
        **{f'Trace {index + 1}': 100 for index in range(len(trace_msg_rank_count))}  
}
        def highlight_cells(val):
            color = ''
            
            if "#BDBDBD" in val: color = '#BDBDBD'
            elif "#82FA58" in val: color = '#82FA58'
            elif "#b0721e" in val: color = '#b0721e'
            elif "#0B610B" in val: color = '#0B610B'
            elif "#2E64FE" in val: color = '#2E64FE'
            elif "#00FFFF" in val: color = '#00FFFF'
            elif "#ffffff" in val: color = '#ffffff'
            elif "#A4A4A4" in val: color = '#A4A4A4'
            elif "#ACFA58" in val: color = '#ACFA58'
            elif "#bb7633" in val: color = '#bb7633'
            elif "#3f7633" in val: color = '#3f7633'
            elif "#0000FF" in val: color = '#0000FF'
            elif "#4aeaff" in val: color = '#4aeaff'
            elif "#ffffff" in val: color = '#ffffff'

            return f'background-color: {color}'
        

        # Streamlit-Anwendung
        st.dataframe(
            df.style.applymap(highlight_cells, subset=['Farbe']),
           # df,
            use_container_width=True,
            column_config={
                "Message Rank": "Message Rank",
                "Bedeutung": "Bedeutung",
                #"Farbe": "Farbe",
                **{f'Trace {index + 1}': f'Anzahl' for index in range(len(trace_msg_rank_count))}
            },
            hide_index=True,
            height=597,
        )
        


        #Tabs für die einzelnen Prozessdarstellungen
        tab1, tab2, tab3 = st.tabs(["BPMN Model", "Heuristic Net", "Trace(s)"])

        with tab1:
            st.header("BPMN Model")
            st.image("bpmn.png")

        with tab2:
            st.header("Heuristic Net")
            st.image("heu_net.png")

        with tab3:
            st.header("Trace(s)")
            st.image("trace.png")    

           

def page3():
    st.title("Predictive Process Monitoring")
    # Hier Inhalt für Seite 2 definieren

def page4():
    def calculate_clusters(df):
        df['NetproJob'] = (df.groupby('Job')['Net'].transform('max') - df.groupby('Job')['Net'].transform('min'))
        df['GrossproJob'] = (df.groupby('Job')['Gross'].transform('max') - df.groupby('Job')['Gross'].transform('min'))

        features = df.groupby('Job').agg({'Speed': 'mean', 'NetproJob': 'first'}).reset_index()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features[['Speed', 'NetproJob']])

        num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters)
        features['cluster'] = kmeans.fit_predict(scaled_features)

        return features

    def visualize_clusters(features):
        plt.scatter(features['Speed'], features['NetproJob'], c=features['cluster'], cmap='viridis')
        plt.title('KMeans Clustering of Job')
        plt.xlabel('Mean Speed')
        plt.ylabel('Net')
        for i, txt in enumerate(features['Job']):
            plt.annotate(txt, (features['Speed'][i], features['NetproJob'][i]), textcoords="offset points", xytext=(0,5), ha='center')

        st.pyplot()

    def main():
        st.title('Clustering')

        # Upload data
        uploaded_cluster_file = st.file_uploader('Upload your data (CSV file)', type='csv')

        if uploaded_cluster_file is not None:
            st.subheader('Uploaded Data Preview:')
            df = pd.read_csv(uploaded_file)
            st.write(df.head())

            # Calculate clusters
            features = calculate_clusters(df)

            # Visualize clusters
            st.subheader('Clustering Results:')
            visualize_clusters(features)

    if __name__ == "__main__":
        main()



# Hauptprogramm
def main():
    st.sidebar.title("Navigation")
    pages = {"Home": home, "KPI": page1, "Process View": page2, "Predictive Process Monitoring": page3}
    selection = st.sidebar.radio("Navigate To", list(pages.keys()))

    # Seiteninhalt anzeigen
    pages[selection]()

if __name__ == "__main__":
    main()
