import pandas as pd
import pm4py

def create_xes_file:
    """ 
	Function to create .xes file from dataset
    """
    # read csv file with data of one machine with job partinioning
    dataset = pd.read_csv('C:/a_Daten/Carolin/Uni/Master/Semester 3/Projektseminar/Daten/MachineD_With_Jobs.csv', sep=';',
                          parse_dates=['CheckIn', 'CheckOut'])


    # create an event log and then a .xes file
    dataset['Job'] = dataset['Job'].astype(str)
    event_log = pm4py.format_dataframe(
      dataset,
      case_id='Job',
      activity_key='MsgID',
      timestamp_key='CheckIn')
    pm4py.write_xes(event_log,
                  'C:/ExamplePath/MachineA_testWithJob-exported.xes')
    return event log

    # Optional: Display the process graph with pm4py
    log = pm4py.read_xes('C:/ExamplePath/MachineA_testWithJob-exported.xes')
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'], format='mixed', errors='coerce')
    process_tree = pm4py.discover_process_tree_inductive(log)
    # create a process model with pm4py and display it graphically
    bpmn_model = pm4py.convert_to_bpmn(process_tree)
    pm4py.view_bpmn(bpmn_model)
