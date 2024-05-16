import pandas as pd
import pm4py

from datetime import timedelta
from collections import namedtuple

def analyse_lg1_messages(dataset: pd.DataFrame, print_phase_change: bool = False, check_sapc: bool = True):
    """ Function to analyze the lg1 messages to define phases.

    Args:
        dataset: Pandas dataframe; containing all basic informations downloaded with load_data.
        print_phase_change: Boolean; if True a phase change is printed in the console.
        check_sapc: Boolean; checks also SAPC.

    # Returns:
        Named tuple with the following informations:
            phase_active_periods: Pandas dataframe; Contains Phase, start, end and MsgID.
            phase_info: Pandas dataframe; Contains Phase, start, end and id.
            fine_tune_sheets: Pandas dataframe: Contains Phase_ID and Sheets.
            net_month: Integer; contains the amount of net for the current month.
            gross_month: Integer; contains the amount of gross for the current month.
    """
    # Start values
    dataset["Phase"] = ""
    new_current_state = 'On'
    current_state = 'On'
    net = dataset.loc[0, "Net"]
    gross = dataset.loc[0, "Gross"]
    current_sheets = gross
    phase_start_date = dataset.loc[0, "CheckIn"]
    phase_number = 0
    range_dataset = dataset.shape[0]
    off_date = None

    # prepare collection parameters
    phase_info = []
    phase_active_periods = []
    fine_tune_sheets = []

    # calculation of first KPIs
    net_month, gross_month = getNetGros(dataset=dataset)
    # net_month = dataset.loc[range_dataset-1, "Net"] - dataset.loc[0, "Net"]
    # gross_month = dataset.loc[range_dataset-1, "Gross"] - dataset.loc[0, "Gross"]

    # Iteration over each entrie in the LG1 file
    for i in range(range_dataset):

        # update current values
        new_net = dataset.loc[i, "Net"]
        new_gross = dataset.loc[i, "Gross"]

        delta_net = new_net - net
        delta_gross = new_gross - gross

        # analyse message and MsgID
        res_check_message = check_message(dataset=dataset, pos=i, check_sapc=check_sapc)

        # special rule if a new message comes up after 5 minutes after the shutdown the machine is on again
        if off_date is not None:
            if off_date > dataset.loc[i, "CheckIn"]:
                continue

        # check if new phase started
        if res_check_message.phase == "Off" and current_state != 'off':
            print_phase_change_fct(print_phase_change=print_phase_change,
                                   new_phase='Maschine off',
                                   i=i,
                                   date=dataset.loc[i, "CheckIn"])
            new_current_state = 'off'
            off_date = dataset.loc[i, "CheckIn"] + timedelta(minutes=5)

        if res_check_message.phase != "Off" and current_state == 'off':
            print_phase_change_fct(print_phase_change=print_phase_change,
                                   new_phase='Maschine on',
                                   i=i,
                                   date=dataset.loc[i, "CheckIn"])
            new_current_state = 'On'

        if res_check_message.phase == "Plate Change" and current_state in ['Other Time', 'Production Time', 'On']:
            print_phase_change_fct(print_phase_change=print_phase_change,
                                   new_phase='Basic Make Ready Time',
                                   i=i,
                                   date=dataset.loc[i, "CheckIn"])
            new_current_state = 'Basic Make Ready Time'

        if current_state in ['Basic Make Ready Time', 'Fine Tune Time', 'On'] and delta_net > 0:
            print_phase_change_fct(print_phase_change=print_phase_change,
                                   new_phase='Production Time',
                                   i=i,
                                   date=dataset.loc[i, "CheckIn"])
            new_current_state = 'Production Time'
        elif current_state in ['Basic Make Ready Time', 'On'] and delta_gross > 0:
            print_phase_change_fct(print_phase_change=print_phase_change,
                                   new_phase='Fine Tune Time',
                                   i=i,
                                   date=dataset.loc[i, "CheckIn"])
            new_current_state = 'Fine Tune Time'

        if current_state in ['Production Time', 'On'] and delta_net > 0:
            possible_end_production = i
            new_current_state = 'Production Time'

        # update phase if phase changed
        if current_state != new_current_state or i == (range_dataset - 1):

            if current_state == 'On':
                # update current state depeding on the new current state value
                if new_current_state == 'Basic Make Ready Time':
                    current_state = 'Other Time'

                if new_current_state == 'Fine Tune Time':
                    current_state = 'Basic Make Ready Time'

                if new_current_state == 'Production Time':
                    current_state = 'Fine Tune Time'

                if new_current_state == 'off' or i == (range_dataset - 1):
                    current_state = 'Other Time'

            # special handling of production time to recreate other time
            if new_current_state != current_state and current_state == 'Production Time':
                phase_info.append([current_state,
                                   phase_start_date,
                                   dataset.loc[possible_end_production, "CheckIn"],
                                   phase_number])

                # write other time status
                if (new_current_state == 'Off' and possible_end_production != i) or new_current_state != 'Off':
                    phase_number = phase_number + 1
                    phase_info.append(['Other Time', dataset.loc[possible_end_production, "CheckIn"],
                                       dataset.loc[i, "CheckIn"], phase_number])

            else:

                # put end of phase before
                phase_info.append([current_state, phase_start_date, dataset.loc[i, "CheckIn"], phase_number])

            # save sheets printed during fine tune time
            if current_state == 'Fine Tune Time':
                fine_tune_sheets.append([phase_number, gross - current_sheets])

            # increase current phase number and current_sheets for prints in fine tune
            phase_number = phase_number + 1
            current_sheets = new_gross

            # update current_state and phase start date
            current_state = new_current_state
            dataset.loc[i, "Phase"] = current_state
            phase_start_date = dataset.loc[i, "CheckIn"]

        net = new_net
        gross = new_gross

        # handling of effective events
        if res_check_message.activephase:
            phase_active_periods.append([res_check_message.phase, dataset.loc[i, "CheckIn"],
                                         dataset.loc[i, "CheckOut"], res_check_message.MsgID])

    # prepare output
    phase_active_periods = pd.DataFrame(phase_active_periods, columns=['Phase', 'start', 'end', 'MsgID'])
    phase_info = pd.DataFrame(phase_info, columns=['Phase', 'start', 'end', 'id'])
    fine_tune_sheets = pd.DataFrame(fine_tune_sheets, columns=['Phase_ID', 'Sheets'])

    Results_LG1_Analysis = namedtuple("Results_LG1_Analysis",
                                      ["phase_active_periods", "phase_info",
                                       "fine_tune_sheets", "net_month",
                                       "gross_month"])
    return Results_LG1_Analysis(phase_active_periods, phase_info, fine_tune_sheets, net_month, gross_month)


def check_message(dataset: pd.DataFrame,
                  pos: int,
                  off_gap: str = '5 minutes',
                  check_sapc: bool = True):
    """ Function to analyze the type of the log message. Showing wich current phase is shown.

    Args:
        dataset: Pandas dataframe; containing all basic informations downloaded with load_data.
        pos: Integer; current position to check.
        off_gap: String; defines the used off_gap for the automatically defines a off-period.
        check_sapc: Boolean; checks also SAPC.


    Returns:
        A namedtuple with the following entries:
            phase: String; showing the current phase.
            activephase: Boolean; showing if the current phase is active.
            MsgID: String; showing the current MsgID.
    """
    phase = ""
    activephase = False
    msg_id = dataset.loc[pos, "MsgID"]
    message = dataset.loc[pos, "Message"]
    checkin = dataset.loc[pos, "CheckIn"]
    checkout = dataset.loc[pos, "CheckOut"]
    message_location = dataset.loc[pos, "LocID1"]
    delta_next_message = dataset.loc[pos, "DeltaNextMessage"]

    # currently known message_locations
    possible_message_locations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 160, 170, 180, 190, 200, 210, 19, 29, 49]

    # Check for Plate Changes
    if msg_id in [2435] and message_location in possible_message_locations or \
            message in ['304.1363', '304.1362', '304.1364', '304.1368', '304.1370'] or \
            (message in ['304.1369'] and check_sapc):
        phase = "Plate Change"
        activephase = True

    if (msg_id in [9927] and message_location in possible_message_locations and check_sapc):

        message_manuell_activ = dataset.loc[(dataset['Message'] == '304.1333') & \
                                            (((dataset['CheckIn'] >= checkin) & (dataset['CheckIn'] <= checkout)) | \
                                             ((dataset['CheckOut'] >= checkin) & (dataset['CheckOut'] <= checkout)) | \
                                             ((dataset['CheckIn'] <= checkin) & (dataset['CheckOut'] >= checkout)) \
                                             )].shape[0] > 0
        message_open_zyl = dataset.loc[(dataset['MsgID'] == 1014) & \
                                       (dataset['LocID1'] == message_location) &
                                       (((dataset['CheckIn'] >= checkin) & (dataset['CheckIn'] <= checkout)) | \
                                        ((dataset['CheckOut'] >= checkin) & (dataset['CheckOut'] <= checkout)) | \
                                        ((dataset['CheckIn'] <= checkin) & (dataset['CheckOut'] >= checkout)) \
                                        )].shape[0] > 0

        if message_manuell_activ & message_open_zyl:
            phase = "Plate Change"
            activephase = True

    if message in ['300.4059', '110.1780', '110.6490'] or delta_next_message > pd.Timedelta(off_gap):
        check_off = check_correct_off(pos=pos, dataset=dataset, wait_for_gap="180 seconds")
        if check_off is True:
            phase = "Off"

    if message in ['300.4059', '110.1780', '110.6490']:
        check_off = check_correct_off(pos=pos, dataset=dataset, wait_for_gap="180 seconds")
        if check_off is True:
            phase = "Off"

    # check effective part T (check for plate change is above)

    # check for Manual Setup
    if message in ['304.1333']:
        activephase = True
        phase = 'check for manual setup'

    # check for engaging/disengaging
    if message in ['304.1365', '304.1366', '304.1367']:
        activephase = True
        phase = 'check for engaging/disengaging'

    # check for Printing is Active
    if message in ['304.1338', '304.1339', '304.1340']:
        activephase = True
        phase = 'check for printing is active'

    # check for Sheet run
    if message in ['304.1342', '304.1341', '304.1343']:
        activephase = True
        phase = 'check for sheet run'

    # check for Perfecting Change Over
    if message in ['304.1360', '304.1359', '304.1361']:
        activephase = True
        phase = 'check for perfecting change over'

    # check for Blanket washing
    if message in ['304.1351', '304.1350', '304.1352']:
        activephase = True
        phase = 'check for blanket washing'

    # check for Impression Cylinder washing
    if message in ['304.1354', '304.1353', '304.1355']:
        activephase = True
        phase = 'check for impression cylinder washing'

    # check for Ink Roller washing
    if message in ['304.1345', '304.1344', '304.1346']:
        activephase = True
        phase = 'check for ink roller washing'

    # check for Plate Cylinder washing
    if message in ['304.1348', '304.1347', '304.1349']:
        activephase = True
        phase = 'check for plate cylinder washing'

    # check for Ink Running in
    if message in ['304.1357', '304.1356', '304.1358']:
        activephase = True
        phase = 'check for ink running in'

    results_check_message = namedtuple("results_check_message", ["phase", "activephase", "MsgID"])
    return results_check_message(phase, activephase, msg_id)


def getNetGros(dataset: pd.DataFrame):
    """ Function to calcualte the gros and net per day to avoid invalid days.

    Args:
        dataset: Pandas dataframe; containing all basic informations downloaded with load_data.

    Returns:
        Two integers; the first is the net month value, the second the gros month value.
    """
    net_month_part = 0
    gross_month_part = 0
    for day in set(dataset["FileDate"]):
        part_dataset = dataset.loc[dataset["FileDate"] == day]
        net_month_part = net_month_part + max(part_dataset["Net"]) - min(part_dataset["Net"])
        gross_month_part = gross_month_part + max(part_dataset["Gross"]) - min(part_dataset["Gross"])

    return net_month_part, gross_month_part


# Analyse
def check_correct_off(pos: int,
                      dataset: pd.DataFrame,
                      wait_for_gap: str = "120 seconds",
                      wait_for_1780: str = "180 seconds",
                      need_gap: str = "300 seconds",
                      wait_for_initial: str = "300 seconds",
                      off_gap: str = "300 seconds") -> bool:
    """ Function to check if an off message defines a correct off period.

    Args:
        pos: Integer; current position to check.
        dataset: Pandas dataframe; containing all basic informations downloaded with load_data.
        wait_for_gap: String; defines the time in which a time gap has to be, to be a correct off period.
        wait_for_1780:  String; defines the waiting time for a 1780 message after a 4059 message.
        need_gap:  String; defines the minimum time gap that has to be exisits to be a correct off period.
        wait_for_initial:  String; defines the time in which an initialisation
                                    has to be done to be a correct off period.
        off_gap:  String; defines the minimum time gap for an off period which is then automatically used as off period.

    Returns:
        Boolean if the given off message is valid.
    """
    start = dataset.loc[pos, "CheckIn"]
    message = dataset.loc[pos, "Message"]
    delta_next_message = dataset.loc[pos, "DeltaNextMessage"]
    res = False

    if message == "300.4059":
        if (dataset.loc[(dataset["CheckIn"] >= start)
                        & (dataset["CheckIn"] <= start + pd.Timedelta(wait_for_1780))
                        & (dataset["Message"] == "110.1780")].shape[0] > 0):
            res = False
    # if an off message is send and in the time period of the next message is an initialise see this as off
    if message in ["110.1780", "110.6490", "300.4059"]:
        gap_data = dataset.loc[(dataset["CheckIn"] >= start) & \
                               (dataset["CheckIn"] <= start + pd.Timedelta(wait_for_gap)) & \
                               (dataset["DeltaNextMessage"] > pd.Timedelta(need_gap))]
        if gap_data.shape[0] > 0:
            pos_gap = gap_data.index[0]
            end_gap = dataset.loc[(pos_gap + 1), "CheckIn"]
            if dataset.loc[(dataset["CheckIn"] >= end_gap) & \
                           (dataset["CheckIn"] <= end_gap + pd.Timedelta(wait_for_initial)) & \
                           (dataset["Message"] == "300.1010")].shape[0] > 0:
                res = True
    # if a long periode without any message see this as off
    if delta_next_message > pd.Timedelta(off_gap):
        res = True

    return res


def print_phase_change_fct(print_phase_change: bool, new_phase: str, i: int, date: str) -> None:
    """ Function for debugging. Shows if a new phase is started.

    Args:
        print_phase_change: Boolean; only if True a print is done.
        new_phase: String; name of the new phase.
        i: Integer; current position.
        date: String; start date of the new phase.

    Returns:
        Nothing. Prints that a new phase is started.

    """
    if print_phase_change:
        print(new_phase + " started at position: " + str(i) + " and date: " + str(date))


def add_jobs(dataset: pd.DataFrame) -> pd.DataFrame:
    """Assigns jobs to the dataset

    Args:
        dataset

    Returns:
        dataset.    New column "Job" is added. All messages that belong to the same job get the same nr.
                    For each new Job the nr. is incremented.
    """

    dataset['Job'] = None
    job_number = 1
    last_net_increase_index = 0

    phase_to_start_new_job = 'Basic Make Ready Time'
    dataset.loc[0, 'Job'] = 1

    # iterate through all lines to add the job numbers
    # the job number is incremented after 'Net' is incremented the last time before the phase 'Basic Make Ready Time' starts
    for i in range(1, len(dataset)):
        current_net = dataset.loc[i, 'Net']
        previous_net = dataset.loc[i - 1, 'Net']
        current_phase = dataset.loc[i, 'Phase']
        if current_phase == phase_to_start_new_job:
            if i != last_net_increase_index:
                if last_net_increase_index == 0:
                    job_number = 1
                else:
                    job_number += 1
                    for j in range(last_net_increase_index + 1, i):
                        dataset.loc[j, 'Job'] = job_number
            last_net_increase_index = i
            dataset.loc[i, 'Job'] = job_number
        elif current_net > previous_net:
            dataset.loc[i, 'Job'] = job_number
            last_net_increase_index = i
        else:
            dataset.loc[i, 'Job'] = job_number
        #dataset.loc[i, 'Phase'] = current_phase

def format_message(message):
    """Formats the 'Message' column uniformly

    Args:
        string/int

    Returns:
        string.    The message is returned in a standardized format as a string
    """
    if isinstance(message, int):
        message = str(message)
    message = message.replace(".", "")
    formatted_message = message[:-4] + '.' + message[-4:]
    return formatted_message


if __name__ == "__main__":
    # read csv file with data of one machine
    dataset = pd.read_csv('C:/a_Daten/Carolin/Uni/Master/Semester 3/Projektseminar/Daten/MachineD.csv', sep=';',
                          parse_dates=['CheckIn', 'CheckOut'])

    # make sure that the Message column always has the correct format
    dataset['Message'] = dataset['Message'].apply(format_message)

    # add column "DeltaNextMessage"
    dataset.sort_values(by=['FileDate', 'SeqNr'], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    dataset['CheckIn'] = pd.to_datetime(dataset['CheckIn'], format='mixed', errors='coerce')
    dataset['CheckOut'] = pd.to_datetime(dataset['CheckOut'], format='mixed', errors='coerce')
    delta = (dataset.loc[1:, "CheckIn"].reset_index(drop=True)
             - dataset.loc[:(dataset.shape[0] - 2), "CheckIn"].reset_index(drop=True))
    dataset['DeltaNextMessage'] = delta

    # analyse all messages and add the job numbering
    analyse_lg1_messages(dataset)
    add_jobs(dataset)

    # save the data frame in a csv file
    output_csv_path = 'C:/a_Daten/Carolin/Uni/Master/Semester 3/Projektseminar/Daten/Neu/MachineD_test.csv'
    dataset.to_csv(output_csv_path, sep=';', index=False)


