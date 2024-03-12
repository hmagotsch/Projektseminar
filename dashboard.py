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
#import plotly.graph_objects as go

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

        log = pm4py.read_xes('seq1and2MachineB-exported.xes')
        log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], format='%Y-%m-%dT%H:%M:%S.%f%z', errors='coerce')

        #Footprints für jede Trace berechnen
        fp_trace_by_trace = footprints_discovery.apply(log, variant=footprints_discovery.Variants.TRACE_BY_TRACE)

        #Extrahiere die relevanten Spalten aus dem Log
        relevant_columns = ['MsgValueDE', 'MsgRank']

        #Entferne Duplikate, um eindeutige Paare von MsgValueDE und MsgRank zu erhalten
        msg_rank_df = log[relevant_columns].drop_duplicates()

        #Mapping für die Farben
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

        # Speichere die Farben, den MsgRank und die Anzahl für jede Trace in trace_dash
        trace_info_list = []

        for trace in fp_trace_by_trace:
            trace_info = pd.DataFrame(trace['trace'], columns=['MsgValueDE'])
            trace_info = pd.merge(trace_info, msg_rank_df, on='MsgValueDE')
            trace_info['MsgValueDECount'] = trace_info.groupby('MsgValueDE')['MsgValueDE'].transform('count')
            trace_info = trace_info[['MsgValueDE', 'MsgRank', 'Color', 'MsgValueDECount']].drop_duplicates()
            trace_info = trace_info.sort_values(by='MsgValueDECount', ascending=False)# Sortiere nach MsgCount absteigend
            trace_info_list.append(trace_info)

        trace_dash2 = [list(trace['trace']) for trace in fp_trace_by_trace]

        anzahl_traces = len(trace_dash2)

        # Anzahl der Einträge in den inneren Listen (Anzahl der Schritte pro Trace)
        anzahl_schritte_pro_trace = [len(inner_list) for inner_list in trace_dash2]

        gesamtanzahl_schritte = sum(anzahl_schritte_pro_trace)

        col1, col2, col3 = st.columns(3)
        col1.metric("Number of print jobs", anzahl_traces)
        col2.metric("Total Number of Messages", gesamtanzahl_schritte)
        col3.metric("Number of Messages per print job", str(anzahl_schritte_pro_trace))
        
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
        st.divider()

        # Erstelle eine leere Liste, um die Top 10 MsgValueDE mit Count zu speichern
        top_10_msgs_list = []

        for trace_info in trace_info_list:
            top_10_msgs = trace_info.head(10)[['MsgValueDE', 'MsgValueDECount']]
            top_10_msgs_list.append(top_10_msgs)

         # Zeige die Liste mit den Top 10 MsgValueDE für jede Trace an
        for index, top_10_msgs in enumerate(top_10_msgs_list):
            st.write(f"Top 10 MsgValueDE for print job {index + 1}:")
            st.dataframe(top_10_msgs, hide_index=True)

        st.divider()
        st.write("Occurence of Messages Ranks:")    

        trace_msg_rank_count = []

        for trace_info in trace_info_list:
            msg_rank_count = trace_info.groupby('MsgRank')['MsgValueDECount'].sum().reset_index()
            trace_msg_rank_count.append(msg_rank_count)
        
        # Erstellen Sie die DataFrame mit den gewünschten Einträgen
        df = pd.DataFrame({
            "Message Rank": ["< 99", -1, 1, 2, 3, 4, 5, 6, "> 98", 99, 101, 102, 103, 104, 105, 106],
            "Color": ["-", "#BDBDBD", "#82FA58", "#b0721e", "#0B610B", "#2E64FE", "#00FFFF", "#ffffff", "-", "#A4A4A4", "#ACFA58", "#bb7633", "#3f7633", "#0000FF", "#4aeaff", "#ffffff"],
            "Meaning": ["durch den Drucker behebbare Meldungen", "noch nicht zugeordnet", "Information", "Warnung", "eingeschränkte Produktion möglich", "keine Produktion möglich", "nicht zuordenbar", "Info nur im Logfile", "durch den Drucker nicht behebare Meldungen", "noch nicht zugeordnet", "Information", "Warnung", "eingeschränkte Produktion möglich", "keine Produktion möglich", "nicht zuordenbar", "Info nur im Logfile"],
        })

        # Iteriere durch jede Trace-Information in trace_msg_rank_count und fülle den DataFrame
        for index, msg_rank_count_df in enumerate(trace_msg_rank_count):
            msg_rank_count_df = msg_rank_count_df.rename(columns={'MsgValueDECount': f'Print job {index + 1}'})
            df = pd.merge(df, msg_rank_count_df, how='left', left_on='Message Rank', right_on='MsgRank')
            df = df.drop(columns=['MsgRank'])

        # Fülle NaN-Werte mit 0
        df = df.fillna(0)

        #Zellen werden mit Hilfe des Mappings farblich hervorgehoben
        def highlight_cells(val):
            color = ''
            for key, value in mapping.items():
                if value in val:
                    color = value
                    break
            return f'background-color: {color}'

        #Anzahl ist immer ganzzahlig und soll daher Integer sein
        def format_number(val):
            try:
                return f'{int(val):,}'
            except ValueError:
                return val

        #Zelle für die beiden Oberbegriffe hat keinen Eintrag
        for i in range(anzahl_traces):
            row_index = 0
            for x in range (2):    
                column_name = f'Print job {i + 1}'  
                new_value = ""
                df.at[row_index, column_name] = new_value
                row_index += 8

        st.dataframe(
            df.style.applymap(highlight_cells, subset=['Color']).format(subset=[f'Print job {index + 1}' for index in range(len(trace_msg_rank_count))], formatter=format_number),
            use_container_width=True,
            column_config={
                "Message Rank": "Message Rank",
                "Meaning": "Meaning",
                **{f'Print job {index + 1}': f'Number Print job {index + 1}' for index in range(len(trace_msg_rank_count))}
            },
            hide_index=True,
            height=606,
        )
        
        st.divider()
        st.write("Process model represenation:")

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
    def calculate_clusters(df, num_clusters):


        #convert CheckIn and CheckOut to datetime
        #df['CheckIn'] = pd.to_datetime(df['CheckIn'])
        #df['CheckOut'] = pd.to_datetime(df['CheckOut'])


        #calculate the duration of each job
        #df['Duration']=(df.groupby('Job')['CheckOut'].transform('max')-df.groupby('Job')['CheckIn'].transform('min')).dt.total_seconds()
        #df['Duration']

        

        df['NetproJob'] = (df.groupby('Job')['Net'].transform('max') - df.groupby('Job')['Net'].transform('min'))
        df['GrossproJob'] = (df.groupby('Job')['Gross'].transform('max') - df.groupby('Job')['Gross'].transform('min'))

        features = df.groupby('Job').agg({'Speed': 'mean', 'NetproJob': 'first','Duration':'first','GrossproJob':'first'}).reset_index()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features[['Speed', 'NetproJob','Duration','GrossproJob']])

        #num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters)
        features['cluster'] = kmeans.fit_predict(scaled_features)

        return features

    def visualize_clusters_plt(features,x_feature, y_feature):
        #Ansatz mit plt
        fig, ax = plt.subplots()
        scatter = ax.scatter(features[x_feature], features[y_feature], c=features['cluster'], cmap='viridis')
        ax.set_title('KMeans Clustering of Job')
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        for i, txt in enumerate(features['Job']):
            ax.annotate(txt, (features[x_feature][i], features[y_feature][i]), textcoords="offset points", xytext=(0,5), ha='center')

        st.pyplot(fig)

    def visualize_clusters_pl(features,x_feature,y_feature):
        #Ansatz mit plotly
        fig = go.Figure()

        for cluster_value in features['cluster'].unique():
            cluster_data = features[features['cluster'] == cluster_value]
            fig.add_trace(go.Scatter(
                x=cluster_data[x_feature],
                y=cluster_data[y_feature],
                mode='markers',
                marker=dict(color=cluster_value),
                text=cluster_data['Job'],
                name=f'Cluster {cluster_value}'
             ))

        fig.update_layout(
            title='KMeans Clustering of Job',
            xaxis=dict(title=x_feature),
            yaxis=dict(title=y_feature),
           showlegend=True
        )

        st.plotly_chart(fig)

    def main():
        st.title('Clustering')

        # Upload data
        uploaded_cluster_file = st.file_uploader('Upload your data (CSV file)', type='csv')

        if uploaded_cluster_file is not None:
            st.subheader('Uploaded Data Preview:')
            df = pd.read_csv(uploaded_cluster_file,sep=",")
            st.write(df.head())

            #Sidebar for user input
            st.sidebar.header('Cluster Settings')
            num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=5)

            #choose features for visualization
            x_feature = st.sidebar.selectbox('Select X Feature', ['Speed', 'NetproJob', 'Duration', 'GrossproJob'])
            y_feature = st.sidebar.selectbox('Select Y Feature', ['Speed', 'NetproJob', 'Duration', 'GrossproJob'])



            # Calculate clusters
            features = calculate_clusters(df,num_clusters)

            # Visualize clusters
            st.subheader('Clustering Results:')
            visualize_clusters_plt(features, x_feature, y_feature)


            visualize_clusters_pl(features,x_feature, y_feature)

            

    if __name__ == "__main__":
        main()



# Hauptprogramm
def main():
    st.sidebar.title("Navigation")
    pages = {"Home": home, "KPI": page1, "Process View": page2, "Predictive Process Monitoring": page3,"Clustering":page4}
    selection = st.sidebar.radio("Navigate To", list(pages.keys()))

    # Seiteninhalt anzeigen
    pages[selection]()

if __name__ == "__main__":
    main()

