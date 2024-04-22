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
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import statsmodels.api as sm
from scipy.stats import t
from typing import List, Tuple
from streamlit_option_menu import option_menu

st.set_page_config(layout='wide')
# Funktionen für verschiedene Seiten
def home():
    st.title("K&B Analytics")

    # First row with two columns
    col1, col2 = st.columns(2)

    # First quadratic section (top left)
    with col1:
        st.header("KPI")
        st.image("https://i.imgur.com/ndWa5kp.jpg")#, width=600, clamp=True)
        expander1 = st.expander("See explanation")
        expander1.write("This page displays key performance indicators for the printing process. Here, you can view the essential data points for each machine and assess the performance of both past and current jobs.")

    # Second quadratic section (top right)
    with col2:
        st.header("Process View")
        st.image("https://i.imgur.com/SN9oJDC.jpg")#, width=200, clamp=True)
        expander2 = st.expander("See explanation")
        expander2.write("The process view page displays the printing process of a job and machine using BPMN notation, leveraging process mining to extract log entries. This allows for the automated generation of process diagrams and in-depth analysis.")

    # Second row with two columns
    col3, col4 = st.columns(2)

    # Third quadratic section (bottom left)
    with col3:
        st.header("Predictive Process Monitoring")
        st.image("https://i.imgur.com/URE8B4M.jpg")
        expander3 = st.expander("See explanation")
        expander3.write("The page for predictive process monitoring enables users to forecast machine errors using an LSTM model. This model examines an input sequence to determine the likelihood of an production stop occurring within the next 10 minutes.")

    # Fourth quadratic section (bottom right)
    with col4:
        st.header("Clustering")
        st.image("https://i.imgur.com/OqspZSv.jpg")
        expander4 = st.expander("See explanation")
        expander4.write("The clustering page employs a K-means algorithm to group printing jobs from different machines based on key performance indicators.")


@st.cache_data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file, sep=';')
        return df
    else:
        return None

@st.cache_data
def calculate_kpis(data: pd.DataFrame) -> List[float]:
      net_values = data['Net']
      num_jobs = data['Job'].max()
      waste = data['Gross'].max() - data['Gross'].max()
      gained_net = net_values.max() - net_values.min()
      avg_speed = round(data['Speed'].mean())
      return [gained_net, waste, num_jobs, avg_speed]

def display_kpi_metrics(kpis: List[float], kpi_names: List[str], units: List[str]):
    st.header("Basic Metrics")
    for i, (col, (kpi_name, kpi_value, unit)) in enumerate(zip(st.columns(4), zip(kpi_names, kpis, units))):
        col.metric(label=kpi_name, value=kpi_value, delta=None)
        col.markdown(f"<p style='font-size: small;'>{unit}</p>", unsafe_allow_html=True)

def page1():
    st.title("Key Performance Indicators")
    file_path = f"/content/MachineA_with_Job.csv"
    #Upload Data
    uploaded_cluster_file_KPI = None

    col1_KPI, col2_KPI = st.columns(2)
    with col1_KPI:
      if uploaded_cluster_file_KPI is None:

        options = ['Machine A', 'Machine B', 'Machine C', 'Machine D']
        selected_option = st.selectbox('Select Machine', options)

        df = load_data(file_path)
        # Display multiselect widget to filter data based on jobs
        all_jobs = sorted(df['Job'].unique())
        selected_jobs = st.multiselect('Select jobs', all_jobs, default=all_jobs)

        if selected_jobs:
          filtered_df = df[df['Job'].isin(selected_jobs)]
          kpis = calculate_kpis(filtered_df)
          kpi_names = ["Netto Sheets", "Waste", "Jobs", "Speed"]
          units= ["[sheets]", "[sheets]", "[number of jobs]", "[sheets/min]"]

        display_kpi_metrics(kpis, kpi_names, units)
        
      tabs = ["Productive and non-productive time per sheet", "Overall Equipment Effectiveness"]
      selected_tab = st.selectbox("Select additional KPI", tabs)

      # Hier Inhalt für Seite 2 definieren
      if selected_tab == "Productive and non-productive time per sheet":

        if uploaded_cluster_file_KPI is None:

          # Read the CSV file into a pandas DataFrame
          df = load_data(file_path)
          import time

          #gewichtete kombinierte Zeit pro Sheet

          # Convert the "CheckIn" and "CheckOut" columns to datetime objects with the specified format
          date_format = "%Y-%m-%d %H:%M:%S%z"
          df['CheckIn'] = df['CheckIn'].str.replace('\.\d+', '', regex=True)
          df['CheckOut'] = df['CheckOut'].str.replace('\.\d+', '', regex=True)
          df['CheckIn'] = pd.to_datetime(df['CheckIn'], format=date_format)
          df['CheckOut'] = pd.to_datetime(df['CheckOut'], format=date_format)
          # Add an additional row where speed is not equal to 0 for the last checkout of the last job
          last_checkout = df[df['Job'] == df['Job'].max()]['CheckIn'].max()

          new_row = pd.DataFrame({'Machine': ["Machine_A"],
                                  'FileDate': [df['FileDate'].iloc[-1]],
                                  'CheckIn': [last_checkout]})
          df = pd.concat([df, new_row], ignore_index=True)


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
          gained_net = 0
          last_occurrences = df.groupby('Job').last()['Net'].to_dict()

          start_time = time.time()
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
                  #gained_net_end = df.loc[(df['Job'] == job)]['Net'].iloc[-1]
                  gained_net_end = last_occurrences.get(job)
                  gained_net = gained_net_end - gained_net_start
                  job_gained_net[job] = gained_net
                  gained_net_start = row['Net']

              # Check if this row belongs to the same job and if it's a later timestamp
              if job == current_job and checkin > production_start_time:
                  job_production_times[current_job] = checkin - production_start_time
                  job_non_productive_times[current_job] = non_productive_time

          # Calculate the job time for each job
          end_time = time.time()
          elapsed_time = end_time - start_time
          print(f"Elapsed time: {elapsed_time} seconds")
          df['JobTime'] = df.groupby('Job')['CheckOut'].transform('max') - df.groupby('Job')['CheckIn'].transform('min')
          job_job_times = df.groupby('Job')['JobTime'].first()

          # Create a list to store the KPI for each job
          kpis = []
          # Print the production time, job time, non-productive time, quotient, gained net, and kpi for each job
          print("Test3")
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

          #Plotly Chart
          #st.markdown("<h3>Productive and non-productive time per sheet <span style='font-size: small; vertical-align: top;' title='The message that is likely to appear next in the current printing process is displayed here. The basis for the prediction is a multinomial bayes model.'>ⓘ</span></h3>", unsafe_allow_html=True)
          x_values = list(range(1, len(kpis) + 1))
          tooltip_text = "This is additional information."
          KPI_figure_2 = go.Figure()
          KPI_figure_2.add_trace(go.Scatter(x=x_values, y=kpis, mode='lines+markers', name='KPI', marker=dict(color='#4A688F')))
          #KPI_figure_2.add_trace(go.Scatter(x=x_values, y=kpis, mode='lines+markers', name='KPI', marker=dict(color='green')))
          KPI_figure_2.update_layout(
              title='Productive and non-productive time per sheet',
              xaxis_title='Job',
              yaxis_title='KPI',
              xaxis=dict(tickvals=x_values, tickmode='linear'),
              yaxis=dict(gridcolor='lightgrey'),
              plot_bgcolor='rgba(255, 255, 255, 0)'  # Transparent background
          )

          # Display the Plotly figure using Streamlit's st.plotly_chart
          st.plotly_chart(KPI_figure_2)

      if selected_tab == "Overall Equipment Effectiveness":

        if uploaded_cluster_file_KPI is None:

          # Read the CSV file into a pandas DataFrame
          df = load_data(file_path)

          # Convert the "CheckIn" and "CheckOut" columns to datetime objects with the specified format
          date_format = "%Y-%m-%d %H:%M:%S%z"
          df['CheckIn'] = df['CheckIn'].str.replace('\.\d+', '', regex=True)
          df['CheckOut'] = df['CheckOut'].str.replace('\.\d+', '', regex=True)
          df['CheckIn'] = pd.to_datetime(df['CheckIn'], format=date_format)
          df['CheckOut'] = pd.to_datetime(df['CheckOut'], format=date_format)
          # Add an additional row where speed is not equal to 0 for the last checkout of the last job
          last_checkout = df[df['Job'] == df['Job'].max()]['CheckIn'].max()

          new_row = pd.DataFrame({'Machine': ["Machine_A"],
                                  'FileDate': [df['FileDate'].iloc[-1]],
                                  'CheckIn': [last_checkout]})
          df = pd.concat([df, new_row], ignore_index=True)

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

              #print(f'Job {job}: A = {A}, P = {P}, Q= {Q}, OEE = {OEE}')

          x_values = list(range(1, len(OEEs) + 1))

          KPI_figure = go.Figure()
          KPI_figure.add_trace(go.Scatter(x=x_values, y=OEEs, mode='lines+markers', name='OEE', marker=dict(color='#4A688F')))
          KPI_figure.update_layout(
              title='Overall Equipment Effectiveness',
              xaxis_title='Job',
              yaxis_title='KPI',
              xaxis=dict(tickvals=x_values, tickmode='linear'),
              yaxis=dict(gridcolor='lightgrey'),
              plot_bgcolor='rgba(255, 255, 255, 0)'  # Transparent background
          )

          st.plotly_chart(KPI_figure)

          #Maschine A - Prediction

          # Read the CSV file into a pandas DataFrame
          #df = pd.read_csv(uploaded_cluster_file_KPI, sep=';')
          #st.write(df.head())
        with col2_KPI:

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
        
            # Scatter plot
            scatter_trace = go.Scatter(
                x=average_speed_per_job,
                y=single_variable,
                mode='markers',
                name='Data points'
            )

            # Regression line
            regression_line = go.Scatter(
                x=average_speed_per_job,
                y=model.predict(X),
                mode='lines',
                name='Regression Line',
                line=dict(color='#E40613')
            )

            # Predicted value marker
            predicted_marker = go.Scatter(
                x=[new_speed_value],
                y=[predicted_value],
                mode='markers',
                name='Predicted Value',
                marker=dict(color='#F39200', size=10)
            )

            # Prediction confidence interval
            ci_line = go.Scatter(
                x=[new_speed_value, new_speed_value],
                y=[prediction_ci[0], prediction_ci[1]],
                mode='lines',
                name='95% CI',
                line=dict(color='#F39200', dash='dash')
            )

            # Layout
            layout = go.Layout(
                title='Scatter Plot with Regression Line and Prediction Interval',
                xaxis=dict(title='Average Speed per Job'),
                yaxis=dict(title='OEE'),
                legend=dict(x=0, y=1)
            )

            # Combine traces and layout into figure
            fig_reg = go.Figure([scatter_trace, regression_line, predicted_marker, ci_line], layout)
            st.plotly_chart(fig_reg)
            st.info(f"The confidence interval of 95% lies between {prediction_ci[0]} and {prediction_ci[1]}.")

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

        st.write("\n")
        
        st.markdown("**General process information:**")
        st.write("\n")

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
        #st.write("\n")

        #col4, col5 = st.columns(2)
        #col4.metric("Startdate", global_start_time.strftime('%Y-%m-%d'))
        #col5.metric("Starttime", global_start_time.strftime('%H:%M:%S.%f'))

        #st.write("\n")
        #st.write("\n")

        #col6, col7 = st.columns(2)
        #col6.metric("Enddate:", global_end_time.strftime('%Y-%m-%d'))
        #col7.metric("Endtime:", global_end_time.strftime('%H:%M:%S.%f'))

        st.divider()
        st.write("\n")


        st.markdown("**Process model represenation:**")

         #Tabs für die einzelnen Prozessdarstellungen
        tab1, tab2, tab3 = st.tabs(["BPMN Model", "Directly-Follows Graph", "Trace(s)"])
        
        with tab1:
            st.header("BPMN Model")
            st.image("bpmn.png")

        with tab2:
            st.header("Directly-Follows Graph")
            st.image("heu_net.png")

        with tab3:
            st.header("Trace(s)")
            st.image("trace.png")


        st.divider()
        st.markdown("**Occurence of Messages Ranks:**")    

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
            height=597,
        )
        
        st.divider()
        # Erstelle eine leere Liste, um die Top 10 MsgValueDE mit Count zu speichern
        top_10_msgs_list = []

        for trace_info in trace_info_list:
            top_10_msgs = trace_info.head(10)[['MsgValueDE', 'MsgValueDECount']]
            top_10_msgs_list.append(top_10_msgs)

        for index in range(0, len(top_10_msgs_list), 2):
            st.markdown(f"**Print job {index + 1} and {index + 2}:**")
            
            # Erstelle zwei Spalten für die Anzeige der Tabellen nebeneinander
            col1, col2 = st.columns(2)

            # Zeige den DataFrame für den ersten Print-Job in der ersten Spalte an
            col1.write(f"Top 10 MsgValueDE for print job {index + 1}:")
            col1.dataframe(top_10_msgs_list[index], hide_index=True)

            # Überprüfe, ob es ein weiteres Element in der Liste gibt, um es in der zweiten Spalte anzuzeigen
            if index + 1 < len(top_10_msgs_list):
                col2.write(f"Top 10 MsgValueDE for print job {index + 2}:")
                col2.dataframe(top_10_msgs_list[index + 1], hide_index=True)


         # Zeige die Liste mit den Top 10 MsgValueDE für jede Trace an
        #for index, top_10_msgs in enumerate(top_10_msgs_list):
        #    st.write(f"Top 10 MsgValueDE for print job {index + 1}:")
        #    st.dataframe(top_10_msgs, hide_index=True)
                

def page3():
    st.title("Predictive Process Monitoring")
    
    machines = {
        "Machine A": {"prediction": True, "message": "Kommunikation mit Positioniereinheit für Seitenanschlag Seite 1 gestört"},
        "Machine B": {"prediction": False, "message": "Rollo im Prozess"},
        "Machine C": {"prediction": True, "message": "Warnung Nonstop wartet auf Hauptstapel herangefahren"},
        "Machine D": {"prediction": False, "message": "Hauptstapel unten (Softwareendlage)"}
    }

    selected_machine = st.selectbox("Select machine:", [""] + list(machines.keys()))

    if selected_machine != "":
        prediction = machines[selected_machine]["prediction"]
        message = machines[selected_machine]["message"]

        st.markdown("<h3>Next message <span style='font-size: small; vertical-align: top;' title='The message that is likely to appear next in the current printing process is displayed here. The basis for the prediction is a multinomial bayes model.'>ⓘ</span>:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div style='border:1px solid #ccc; padding:10px; border-radius:5px;'>{message}</div>", unsafe_allow_html=True)

        st.markdown("<h3>Prediction for production stop <span style='font-size: small; vertical-align: top;' title='Here we predict whether a production stop is likely to occur in the next 5 minutes or not. The associated LSTM model has learned from old data and takes messages with a MsgRank of 104 as indicators for a production stop. Depending on the prediction, the current process can be monitored more closely in order to avoid possible upcoming production stops.'>ⓘ</span>:</h3>", unsafe_allow_html=True)

        if prediction:
            st.error("It is very likely that there will be a production stop in the next 5 minutes.")
            st.write("### ❌")  # Red cross for production stop
        else:
            st.success("It is very unlikely that there will be a production stop in the next 5 minutes.")
            st.write("### ✅")  # Green checkmark for no production stop

        if prediction:
            st.subheader("More information:")
            if st.button("Show more information"):
                # Load data from CSV
                data = pd.read_csv('MachineB_with_Job.csv', sep=';')
                # Filter and display last 10 rows with selected columns
                st.write("### The last 10 Messages")
                st.write(data[['MsgValueDE', 'CheckIn']].tail(10).reset_index(drop=True))
def page4():

    def calculate_clusters(df, num_clusters):


        df['NetproJob'] = (df.groupby('Job')['Net'].transform('max') - df.groupby('Job')['Net'].transform('min'))
        df['GrossproJob'] = (df.groupby('Job')['Gross'].transform('max') - df.groupby('Job')['Gross'].transform('min'))

        features = df.groupby('Job').agg({'Speed': 'mean', 'NetproJob': 'first','Duration':'first','GrossproJob':'first'}).reset_index()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features[['Speed', 'NetproJob','Duration','GrossproJob']])

        #num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters)
        features['cluster'] = kmeans.fit_predict(scaled_features)

        return features
    
    def calculate_clusters_m(df_m, num_clusters_m):
        df_m=df_m[df_m['Total_Jobs']>1]
        features_m = df_m[['Machine','Mean_Speed','Total_Jobs','Net','Gross']]
        scaler = StandardScaler()
        scaled_features_m = scaler.fit_transform(features_m[['Mean_Speed', 'Total_Jobs','Net','Gross']])
        kmeans = KMeans(n_clusters=num_clusters_m)
        features_m['cluster'] = kmeans.fit_predict(scaled_features_m)

        return features_m



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


    def visualize_clusters_3D(features, x3_feature, y3_feature, z3_feature):
        custom_colors = ['#093D79', '#4A688F', '#8BBCE4', '#9D9E9E', '#FFFFFF', '#E40613', '#FA6F7C', '#E4A7AA', '#F39200', '#E1BC89']
    
        fig = go.Figure()

        for cluster_value, color in zip(features['cluster'].unique(), custom_colors):
            cluster_data = features[features['cluster'] == cluster_value]
            fig.add_trace(go.Scatter3d(
                x=cluster_data[x3_feature],
                y=cluster_data[y3_feature],
                z=cluster_data[z3_feature],
                mode='markers', 
                marker=dict(color=color, size=10),
                text=cluster_data['Job'],
                name=f'Cluster {cluster_value}'
             ))

        fig.update_layout(
            title='KMeans Clustering of Job',
            scene=dict(
                xaxis=dict(title=x3_feature),
                yaxis=dict(title=y3_feature),
                zaxis=dict(title=z3_feature)
            ),
            showlegend=True,
            plot_bgcolor='rgba(255, 255, 255, 0)'  # Transparent background
        )

        st.plotly_chart(fig)


    def visualize_clusters_pl(features,x_feature,y_feature):
        #Ansatz mit plotly
        custom_colors = ['#093D79', '#4A688F','#8BBCE4','#9D9E9E','#FFFFFF','#E40613','#FA6F7C','#E4A7AA','#F39200','#E1BC89']
    
        fig = go.Figure()

        for cluster_value, color in zip(features['cluster'].unique(), custom_colors):
            cluster_data = features[features['cluster'] == cluster_value]
            fig.add_trace(go.Scatter(
                x=cluster_data[x_feature],
                y=cluster_data[y_feature],
                mode='markers',
                #marker=dict(color=cluster_value),
                marker=dict(color=color, size=10),
                text=cluster_data['Job'],
                name=f'Cluster {cluster_value}'
                
             ))

        fig.update_layout(
            title='KMeans Clustering of Job',
            xaxis=dict(title=x_feature),
            yaxis=dict(title=y_feature),
           showlegend=True,
           plot_bgcolor='rgba(255, 255, 255, 0)'  # Transparent background
        )

        st.plotly_chart(fig)

      
    def visualize_clusters_pl_m(features_m,xm_feature,ym_feature):
        #Ansatz mit plotly
        #Ansatz mit plotly
        custom_colors = ['#093D79', '#4A688F','#8BBCE4','#9D9E9E','#FFFFFF','#E40613','#FA6F7C','#E4A7AA','#F39200','#E1BC89']
        fig = go.Figure()

       


        for cluster_value, color in zip(features_m['cluster'].unique(), custom_colors):
           cluster_data = features_m[features_m['cluster'] == cluster_value]
           fig.add_trace(go.Scatter(
                x=cluster_data[xm_feature],
                y=cluster_data[ym_feature],
                mode='markers',
                marker=dict(color=color, size=10),
                text=cluster_data['Machine'],
                name=f'Cluster {cluster_value}'
             ))

        fig.update_layout(
            title='KMeans Clustering of Machines',
            xaxis=dict(title=xm_feature),
            yaxis=dict(title=ym_feature),
           showlegend=True,
           legend_traceorder='normal'

        )

        st.plotly_chart(fig)
    

    def visualize_clusters_3D_m_(features_m, x3m_feature, y3m_feature, z3m_feature):

        #custom_colors = ['#093D79', '#4A688F', '#8BBCE4', '#D1D4D4', '#FFFFFF', '#E40613', '#FA6F7C', '#E4A7AA', '#F39200', '#E1BC89']
        
        #fig = px.scatter_3d(features_m, x=x3m_feature, y=y3m_feature, z=z3m_feature, color='cluster', hover_name='Machine')


        #fig.update_layout(title='KMeans Clustering of Machines', scene=dict(xaxis_title=x3m_feature, yaxis_title=y3m_feature, zaxis_title=z3m_feature))
        #st.plotly_chart(fig)
        custom_colors = ['#093D79', '#4A688F', '#8BBCE4', '#D1D4D4', '#FFFFFF', '#E40613', '#FA6F7C', '#E4A7AA', '#F39200', '#E1BC89']

        # Assign a color to each cluster
        cluster_colors = {cluster: color for cluster, color in zip(features_m['cluster'].unique(), custom_colors)}

        fig = px.scatter_3d(features_m, x=x3m_feature, y=y3m_feature, z=z3m_feature, color='cluster', color_discrete_map=cluster_colors, hover_name='Machine')

        fig.update_layout(title='KMeans Clustering of Machines', scene=dict(xaxis_title=x3m_feature, yaxis_title=y3m_feature, zaxis_title=z3m_feature))
        st.plotly_chart(fig)



    def visualize_clusters_3D_m(features_m, x3m_feature, y3m_feature, z3m_feature):
        custom_colors = ['#093D79', '#4A688F', '#8BBCE4', '#9D9E9E', '#FFFFFF', '#E40613', '#FA6F7C', '#E4A7AA', '#F39200', '#E1BC89']
    
        fig = go.Figure()

        for cluster_value, color in zip(features_m['cluster'].unique(), custom_colors):
            cluster_data = features_m[features_m['cluster'] == cluster_value]
            fig.add_trace(go.Scatter3d(
                x=cluster_data[x3m_feature],
                y=cluster_data[y3m_feature],
                z=cluster_data[z3m_feature],
                mode='markers',
                marker=dict(color=color, size=10),
                text=cluster_data['Machine'],
                name=f'Cluster {cluster_value}'
            ))

        fig.update_layout(
            title='KMeans Clustering of Machines',
            scene=dict(
                xaxis=dict(title=x3m_feature),
                yaxis=dict(title=y3m_feature),
                zaxis=dict(title=z3m_feature)
            ),
            showlegend=True,
            plot_bgcolor='rgba(255, 255, 255, 0)'  # Transparent background
        )

        st.plotly_chart(fig)


    def main():
        
        st.title('Clustering')
        #st.subheader('Clustering of Jobs')
        st.markdown("<h3>Clustering of Jobs<span style='font-size:small; vertical-align:top;' title='All the jobs of one machine are clustered'>🛈</span>:</h3>", unsafe_allow_html=True)
                

        #enter path here!!!
        file_path= "c:\\Users\\1412a\\Documents\\Projektseminar Master 2023\\VS_DB\\Maschine D mit Duration.csv"
        uploaded_cluster_file=pd.read_csv(file_path,sep=",")

        if uploaded_cluster_file is not None:
            options = ['Machine A', 'Machine B', 'Machine C', 'Machine D']
            selected_option = st.selectbox('Select Machine', options , index=options.index('Machine D'))
            #st.subheader('Uploaded Data Preview:')
            df=uploaded_cluster_file
            #st.write(df.head())

        
            num_clusters = st.slider('Number of Clusters', min_value=2, max_value=10, value=5, key='slider_job')

             # Calculate clusters
            features = calculate_clusters(df,num_clusters)

            #Split layout into two columns
            left_column, right_column = st.columns(2)

            
            
            with left_column:

                

                st.subheader('Clustering Results (2D):') 
                #st.markdown("<h3>Clustering Results (2D) <span style='font-size:small; vertical-align:top;' title='Clustering Results'>🛈</span>:</h3>", unsafe_allow_html=True)
                

                st.markdown("Settings:")              
                x_feature = st.selectbox('Select X Feature', ['Speed', 'NetproJob', 'Duration', 'GrossproJob'], key='x_feature')
                y_feature = st.selectbox('Select Y Feature', ['Speed', 'NetproJob', 'Duration', 'GrossproJob'], key='y_feature')
                
                
                visualize_clusters_pl(features,x_feature, y_feature)
                st.markdown("<h5>  <span style='font-size:small; vertical-align:top;' title='Hover over the datapoints for more details'>🛈</span>:</h5>", unsafe_allow_html=True)
                
                

            right_column.subheader('Clustering Results (3D)')

            with right_column:
                st.markdown('Settings:')
                x3_feature = st.selectbox('Select X Feature', ['Speed', 'NetproJob', 'Duration', 'GrossproJob'],key='x3_feature')
                y3_feature = st.selectbox('Select Y Feature', ['Speed', 'NetproJob', 'Duration', 'GrossproJob'], key= 'y3_feature')
                z3_feature = st.selectbox('Select Z Feature',['Speed', 'NetproJob', 'Duration', 'GrossproJob'], key='z3_feature')

                visualize_clusters_3D(features,x3_feature,y3_feature,z3_feature)
                




        st.subheader('Clustering of Machines') 

        #file_path2= "c:\\Users\\1412a\\Documents\\Projektseminar Master 2023\\VS_DB\\Data CL_Machines.csv"
        file_path2="https://github.com/hmagotsch/Dashboard/blob/main/Data%20Clustering_Machines.csv"
        uploaded_cluster_file_m=pd.read_csv(file_path2,sep=",")

        # Upload data
        #uploaded_cluster_file_m = st.file_uploader('Upload your data (CSV file)', type='csv',key='cluster_data_machines')

        if uploaded_cluster_file_m is not None:
            #st.subheader('Uploaded Data Preview:')
            #df_m = pd.read_csv(uploaded_cluster_file_m,sep=",")
            df_m = uploaded_cluster_file_m
            #st.write(df_m.head())

            num_clusters_m = st.slider('Number of Clusters', min_value=2, max_value=10, value=5, key='slider_machine')


            # Calculate clusters
            features_m = calculate_clusters_m(df_m,num_clusters_m)

            left_column_m, right_column_m = st.columns(2)
            # Visualize clusters
            left_column_m.subheader('Clustering Results (2D):')
            #visualize_clusters_plt(features, x_feature, y_feature)

            with left_column_m:

                st.markdown('Settings:')
                xm_feature = st.selectbox('Select X Feature', ['Mean_Speed', 'Net', 'Total_Jobs', 'Gross'], key='xm_feature')
                ym_feature = st.selectbox('Select Y Feature', ['Mean_Speed', 'Net', 'Total_Jobs', 'Gross'], key='ym_feature')

                visualize_clusters_pl_m(features_m,xm_feature, ym_feature)

            right_column_m.subheader('Clustering Results (3D)')


            with right_column_m:
                st.markdown('Settings')
                x3m_feature = st.selectbox('Select X Feature', ['Mean_Speed', 'Net', 'Total_Jobs', 'Gross'],key='x3m_feature')
                y3m_feature = st.selectbox('Select Y Feature', ['Mean_Speed', 'Net', 'Total_Jobs', 'Gross'], key= 'y3m_feature')
                z3m_feature = st.selectbox('Select Z Feature',['Mean_Speed', 'Net', 'Total_Jobs', 'Gross'], key='z3m_feature')

                visualize_clusters_3D_m(features_m,x3m_feature,y3m_feature,z3m_feature)

    if __name__ == "__main__":
        main()


@st.cache(allow_output_mutation=True)
def get_pages():
    return {"Home": home, "KPI": page1, "Process View": page2, "Predictive Process Monitoring": page3, "Clustering": page4}

# Hauptprogramm
def main():
#     #alte Variante mit Navigation links
    
#     st.sidebar.title("Navigation")
#     pages = {"Home": home, "KPI": page1, "Process View": page2, "Predictive Process Monitoring": page3,"Clustering":page4}
#     selection = st.sidebar.radio("Navigate To", list(pages.keys()))

#     #Seiteninhalt anzeigen
#     pages[selection]()

# if __name__ == "__main__":
#    main()


  selected = option_menu(
      menu_title=None,  # required
      options=["Home", "KPI", "Process View", "Predictive Process Monitoring", "Clustering"],  # required
      icons=["house", "bar-chart", "diagram-3", "calendar-date", "grid-3x3-gap"],  # optional
      menu_icon="cast",  # optional
      default_index=0,  # optional
      orientation="horizontal",
  )
  #Seiteninhalt anzeigen
  pages = {"Home": home, "KPI": page1, "Process View": page2, "Predictive Process Monitoring": page3,"Clustering":page4}

  if selected == "Home":
      pages["Home"]()
  if selected == "KPI":
      pages["KPI"]()
  if selected == "Process View":
      pages["Process View"]()
  if selected == "Predictive Process Monitoring":
      pages["Predictive Process Monitoring"]()
  if selected == "Clustering":
      pages["Clustering"]()



if __name__ == "__main__":
  main()
