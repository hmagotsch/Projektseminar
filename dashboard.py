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

    # Upload data
    uploaded_cluster_file_KPI = st.file_uploader('Upload your data (CSV file)', type='csv')
    # Hier Inhalt für Seite 2 definieren

    if uploaded_cluster_file_KPI is not None:
      #Maschine A - KPI
      import numpy as np
      import statsmodels.api as sm
      from scipy.stats import t

      # Read the CSV file into a pandas DataFrame
      df = pd.read_csv(uploaded_cluster_file_KPI, sep=';')

      #gewichtete kombinierte Zeit pro Sheet

      # Read the CSV file into a pandas DataFrame

      # Convert the "CheckIn" and "CheckOut" columns to datetime objects with the specified format
      date_format = "%Y-%m-%d %H:%M:%S.%f%z"
      df['CheckIn'] = pd.to_datetime(df['CheckIn'], format=date_format)
      df['CheckOut'] = pd.to_datetime(df['CheckOut'], format=date_format)

      # Add an additional row where speed is not equal to 0 for the last checkout of the last job
      last_checkout = df[df['Job'] == df['Job'].max()]['CheckOut'].max()
      df = df.append({'Machine': df['Machine'].iloc[-1],
                      'FileDate': df['FileDate'].iloc[-1],
                      'CheckIn': last_checkout,
                      'CheckOut': last_checkout,
                      'Speed': 1,  # Change to the appropriate non-zero speed value
                      'Gross': df['Gross'].iloc[-1],
                      'Net': df['Net'].iloc[-1],
                      'MsgRank': df['MsgRank'].iloc[-1],
                      'MsgValueDE': df['MsgValueDE'].iloc[-1],
                      'Phase': df['Phase'].iloc[-1],
                      'Job': df['Job'].max()},
                    ignore_index=True)


      # Initialize dictionaries to keep track of the production time, job time, and non-productive time for each job
      job_production_times = {}
      job_job_times = {}
      job_non_productive_times = {}
      job_gained_net = {}

      current_job = 0
      production_start_time = None
      non_productive_start_time = None
      non_productive_time = pd.Timedelta(0)
      gained_net_start = 0

      for index, row in df.iterrows():
          phase = row['Phase']
          job = row['Job']
          checkin = row['CheckIn']
          checkout = df.at[index - 1, 'CheckOut'] if index > 0 else None
          speed = row['Speed']

          if phase == 'Production Time' and job != current_job:
              # When 'Production Time' is encountered, store the CheckIn timestamp as the start time
              production_start_time = checkin
              current_job = job

              # Reset non-productive time variables
              non_productive_start_time = None
              non_productive_time = pd.Timedelta(0)

          #Phase off is not supposed to be added to production time
          if phase == 'off':
              while index + 1 < len(df) and df.iloc[index + 1]['Phase'] != 'Production Time':
                  index += 1
              if index + 1 < len(df):
                  row = df.iloc[index + 1]
                  phase = row['Phase']
              else:
                  break
              continue


          if speed == 0:
              # When 'Speed' is zero, mark it as a non-productive time period
              if non_productive_start_time is None:
                  non_productive_start_time = checkin

          else:
              # If 'Speed' is not zero and there was a non-productive time, add it to the non-productive time for the job
              if non_productive_start_time is not None:
                  non_productive_time += checkout - non_productive_start_time
                  non_productive_start_time = None

          if job != current_job:
              # Job has changed, calculate the gained net
              gained_net_end = df.loc[(df['Job'] == job)]['Net'].iloc[-1]
              gained_net = gained_net_end - gained_net_start
              job_gained_net[job] = gained_net
              gained_net_start = row['Net']

          # Check if this row belongs to the same job and if it's a later timestamp
          if job == current_job and checkin > production_start_time:
              job_production_times[current_job] = checkin - production_start_time
              job_non_productive_times[current_job] = non_productive_time

      # Calculate the job time for each job
      df['JobTime'] = df.groupby('Job')['CheckOut'].transform('max') - df.groupby('Job')['CheckIn'].transform('min')
      job_job_times = df.groupby('Job')['JobTime'].first()

      # Create a list to store the KPI for each job
      kpis = []

      # Print the production time, job time, non-productive time, quotient, gained net, and kpi for each job
      for job in job_production_times.keys():
          production_time = job_production_times.get(job, pd.Timedelta(0))
          job_time = job_job_times.get(job, pd.Timedelta(0))
          non_productive_time = job_non_productive_times.get(job, pd.Timedelta(0))
          gained_net = job_gained_net.get(job, 0)

          if job_time.total_seconds() == 0:
              quotient = None
          else:
              unproductive_per_net = non_productive_time.total_seconds() / gained_net
              productive_per_net = (production_time.total_seconds() - non_productive_time.total_seconds()) / gained_net
              kpi = 4.23 * unproductive_per_net + productive_per_net
              kpis.append(kpi)

          print(f'Job {job}: Production Time = {production_time}, Non-Productive Time = {non_productive_time}, unproductive_per_net = {unproductive_per_net},productive_per_net = {productive_per_net}, Gained Net = {gained_net}, KPI = {kpi}')

      # Create a line chart for KPI
      KPI_figure_2 = plt.figure(figsize=(10, 6))
      plt.plot(range(1, len(kpis) + 1), kpis, color='green', label='KPI', marker='o')
      plt.xlabel('Job')
      plt.ylabel('KPI')
      plt.title('KPI1 for each Job - Machine A')
      plt.xticks(range(1, len(kpis) + 1))
      plt.grid(True)
      plt.show()

      st.pyplot(KPI_figure_2)

      # Convert the "CheckIn" and "CheckOut" columns to datetime objects with the specified format
      date_format = "%Y-%m-%d %H:%M:%S.%f%z"
      df['CheckIn'] = pd.to_datetime(df['CheckIn'], format=date_format)
      df['CheckOut'] = pd.to_datetime(df['CheckOut'], format=date_format)

      # Add an additional row where speed is not equal to 0 for the last checkout of the last job
      last_checkout = df[df['Job'] == df['Job'].max()]['CheckOut'].max()
      df = df.append({'Machine': df['Machine'].iloc[-1],
                      'FileDate': df['FileDate'].iloc[-1],
                      'CheckIn': last_checkout,
                      'CheckOut': last_checkout,
                      'Speed': 1,  # Change to the appropriate non-zero speed value
                      'Gross': df['Gross'].iloc[-1],
                      'Net': df['Net'].iloc[-1],
                      'MsgRank': df['MsgRank'].iloc[-1],
                      'MsgValueDE': df['MsgValueDE'].iloc[-1],
                      'Phase': df['Phase'].iloc[-1],
                      'Job': df['Job'].max()},
                  ignore_index=True)


      # Initialize dictionaries and variables
      job_production_times = {}
      job_job_times = {}
      job_non_productive_times = {}
      job_gained_net = {}
      job_gained_gross = {}
      job_basic_make_ready_times = {}
      job_fine_tune_times = {}
      job_off_times_pre = {}
      job_off_times_prod = {}
      job_max_speed = {}
      current_job = 0
      current_phase = None
      production_start_time = None
      off_start_time_pre = None
      off_start_time_prod = None
      non_productive_start_time = None
      non_productive_time = pd.Timedelta(0)
      gained_net_start = 0
      gained_gross_start = 0
      job_fine_tune_times = {}
      job_basic_make_ready_times = {}
      fine_tune_start_time = None
      basic_make_ready_start_time = None
      fine_tune_time = pd.Timedelta(0)
      basic_make_ready_time = pd.Timedelta(0)
      off_time_pre = pd.Timedelta(0)
      off_time_prod = pd.Timedelta(0)
      new_speed_value = 0  # You can set any default value here

      #Determine gained gross and net value for each job
      for job in df['Job'].unique():
          job_data = df[df['Job'] == job]

          # Calculate gained net for the job
          net_start = job_data['Net'].min()
          net_end = job_data['Net'].max()
          gained_net = net_end - net_start
          job_gained_net[job] = gained_net

          # Calculate gained gross for the job
          gross_start = job_data['Gross'].min()
          gross_end = job_data['Gross'].max()
          gained_gross = gross_end - gross_start
          job_gained_gross[job] = gained_gross

          #Maximum Speed for each job
          Max_speed = job_data['Speed'].max()
          job_max_speed[job] = Max_speed

      #Loop over whole csv file
      for index, row in df.iterrows():
          phase = row['Phase']
          job = row['Job']
          checkin = row['CheckIn']
          checkout = df.at[index - 1, 'CheckOut'] if index > 0 else None
          speed = row['Speed']

      #Production Time

          if phase == 'Production Time' and job != current_job:
              # When 'Production Time' is encountered, store the CheckIn timestamp as the start time
              production_start_time = checkin
              current_job = job

              # Reset non-productive time variables
              non_productive_start_time = None
              non_productive_time = pd.Timedelta(0)

      #non-productive Time
          if speed == 0:
              # When 'Speed' is zero, mark it as a non-productive time period
              if non_productive_start_time is None:
                  non_productive_start_time = checkin

          else:
              # If 'Speed' is not zero and there was a non-productive time, add it to the non-productive time for the job
              if non_productive_start_time is not None:
                  non_productive_time += checkin - non_productive_start_time
                  non_productive_start_time = None

      #Basic Make Ready Time
          if phase == 'Basic Make Ready Time':
              if basic_make_ready_start_time is None:
                  basic_make_ready_start_time = checkin
                  fine_tune_time = pd.Timedelta(0)
                  off_time_pre = pd.Timedelta(0)
                  off_time_prod = pd.Timedelta(0)

          if phase == 'Fine Tune Time' or phase == 'off':
              if basic_make_ready_start_time is not None:
                  basic_make_ready_time = checkin - basic_make_ready_start_time
                  basic_make_ready_start_time = None
      #Fine Tune Time
          if phase == 'Fine Tune Time':
              if fine_tune_start_time is None:
                  fine_tune_start_time = checkin

          if (phase == 'Production Time' or phase == 'off') and current_phase == 'Fine Tune Time':
              if fine_tune_start_time is not None:
                  fine_tune_time += checkin - fine_tune_start_time
                  fine_tune_start_time = None

      #Off Time in Preprocessing
          if phase == 'off' and (current_phase == 'Fine Tune Time' or current_phase == 'Basic Make Ready Time') :
              if off_start_time_pre is None:
                  off_start_time_pre = checkin

          if phase == 'On' and current_phase == 'off':
              if off_start_time_pre is not None:
                  off_time_pre += checkin - off_start_time_pre
                  off_start_time_pre = None

      #Off Time in Production
          if phase == 'off' and current_phase == 'Production Time':
              if off_start_time_prod is None:
                  off_start_time_prod = checkin

          if phase == 'On' and current_phase == 'off':
              if off_start_time_prod is not None:
                  off_time_prod += checkin - off_start_time_prod
                  off_start_time_prod = None

          if pd.notnull(phase):
                  current_phase = phase


          # Check if this row belongs to the same job and if it's a later timestamp
          if job == current_job and checkin > production_start_time:
              job_production_times[current_job] = checkin - production_start_time
              job_non_productive_times[current_job] = non_productive_time
              job_basic_make_ready_times[current_job] = basic_make_ready_time
              job_fine_tune_times[current_job] = fine_tune_time
              job_off_times_pre[current_job] = off_time_pre
              job_off_times_prod[current_job] = off_time_prod

      # Calculate the job time for each job
      df['JobTime'] = df.groupby('Job')['CheckOut'].transform('max') - df.groupby('Job')['CheckIn'].transform('min')
      job_job_times = df.groupby('Job')['JobTime'].first()

      # Create a list to store the KPI for each job
      As = []
      Ps = []
      Qs = []
      OEEs = []

      # Print the production time, job time, non-productive time, quotient, gained net, and kpi for each job
      for job in job_production_times.keys():
          production_time = job_production_times.get(job, pd.Timedelta(0))
          non_productive_time = job_non_productive_times.get(job, pd.Timedelta(0))
          basic_make_ready_time = job_basic_make_ready_times.get(job, pd.Timedelta(0))
          fine_tune_time = job_fine_tune_times.get(job, pd.Timedelta(0))
          job_time = job_job_times.get(job, pd.Timedelta(0))
          gained_net = job_gained_net.get(job, 0)
          gained_gross = job_gained_gross.get(job, 0)
          off_time_pre = job_off_times_pre.get(job, 0)
          off_time_prod = job_off_times_prod.get(job, 0)
          Max_speed = job_max_speed.get(job, 0)

          A = (production_time.total_seconds() - non_productive_time.total_seconds() - off_time_prod.total_seconds())/ (production_time.total_seconds() + fine_tune_time.total_seconds() + basic_make_ready_time.total_seconds() - off_time_prod.total_seconds())
          P = ((3600/Max_speed) *gained_gross) / (production_time.total_seconds() - non_productive_time.total_seconds() - off_time_prod.total_seconds())
          Q = gained_net / gained_gross
          OEE = A * P * Q

          As.append(A)
          Ps.append(P)
          Qs.append(Q)
          OEEs.append(OEE)

          print(f'Job {job}: A = {A}, P = {P}, Q= {Q}, OEE = {OEE}')

      # Create a line chart for KPI
      # KPI_figure = plt.figure(figsize=(10, 6))
      # plt.plot(range(1, len(OEEs) + 1), OEEs, color='green', label='OEE', marker='o')
      # plt.xlabel('Job')
      # plt.ylabel('OEE')
      # plt.title('OEE for each Job - Machine A')
      # plt.xticks(range(1, len(OEEs) + 1))
      # plt.grid(True)
      # plt.show()

     # Assuming OEEs is your data
      x_values = list(range(1, len(OEEs) + 1))

      KPI_figure = go.Figure()
      KPI_figure.add_trace(go.Scatter(x=x_values, y=OEEs, mode='lines+markers', name='OEE', marker=dict(color='green')))
      KPI_figure.update_layout(title='OEE for each Job - Machine A',
                        xaxis_title='Job',
                        yaxis_title='OEE',
                        xaxis=dict(tickvals=x_values, tickmode='linear'),
                        yaxis=dict(gridcolor='lightgrey'),
                        plot_bgcolor='white')

      st.plotly_chart(KPI_figure)

      #Maschine A - Prediction

      # Read the CSV file into a pandas DataFrame
      #df = pd.read_csv(uploaded_cluster_file_KPI, sep=';')
      #st.write(df.head())

      new_speed_value = st.number_input("Enter New Speed Value:", value=new_speed_value)


      def lin_reg():
        single_variable = OEEs
        average_speed_per_job = df.groupby('Job')['Speed'].mean()

        # Replace these with your actual data
        single_variable = np.array(single_variable)
        average_speed_per_job = np.array(average_speed_per_job)

        # Add a constant term for the intercept
        X = sm.add_constant(average_speed_per_job)

        # Fit the linear regression model with average_speed_per_job as the independent variable
        model = sm.OLS(single_variable, X).fit()

        # Get the slope and intercept from the regression model
        slope = model.params[1]
        intercept = model.params[0]

        # Calculate the prediction for a new value of average_speed_per_job
        # Create an input box for the user to input the value of new_speed_value
        #new_speed_value = st.number_input("Enter New Speed Value:", value=new_speed_value)
        predicted_value = slope * new_speed_value + intercept

        # Get the standard error of the prediction
        se = np.sqrt(model.scale)  # Standard error of the residuals

        # Calculate the confidence interval for the prediction
        alpha = 0.05  # Significance level
        t_value = t.ppf(1 - alpha / 2, df=model.df_resid)  # T-distribution critical value
        margin_of_error = t_value * se
        prediction_ci = (predicted_value - margin_of_error, predicted_value + margin_of_error)

        # Create a scatter plot
        fig, ax = plt.subplots()
        ax.scatter(average_speed_per_job, single_variable, label='Data points')

        # Plot the regression line
        regression_line = model.predict(X)
        ax.plot(average_speed_per_job, regression_line, color='red', label='Regression Line')

        # Highlight the new value and its prediction with the correct confidence interval
        ax.scatter(new_speed_value, predicted_value, color='green', label='Predicted Value')
        ax.plot([new_speed_value, new_speed_value], [prediction_ci[0], prediction_ci[1]], color='green', linestyle='--', label='95% CI')

        # Add labels and a title
        ax.set_xlabel('Average Speed per Job')
        ax.set_ylabel('OEE')
        ax.set_title('Scatter Plot with Regression Line and Prediction Interval')

        # Show the plot
        ax.legend()
        st.pyplot(fig)

        st.write(f"The confidence interval of 95% lies between {prediction_ci[0]} and {prediction_ci[1]}.")

      if st.button("See Linear Regression"):
        # When the button is clicked, call your function
        lin_reg()

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

