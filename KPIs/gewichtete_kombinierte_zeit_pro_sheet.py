
#Maschine A
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('MachineA_with_Job.csv', sep=';')

# Convert the "CheckIn" and "CheckOut" columns to datetime objects with the specified format
date_format = "%Y-%m-%d %H:%M:%S%z"
df['CheckIn'] = df['CheckIn'].str.replace('\.\d+', '', regex=True)
df['CheckOut'] = df['CheckOut'].str.replace('\.\d+', '', regex=True)
df['CheckIn'] = pd.to_datetime(df['CheckIn'], format=date_format)
df['CheckOut'] = pd.to_datetime(df['CheckOut'], format=date_format)

# Add an additional row where speed is not equal to 0 for the last checkout of the last job
last_checkout = df[df['Job'] == df['Job'].max()]['CheckOut'].max()
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
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(kpis) + 1), kpis, color='green', label='KPI', marker='o')
plt.xlabel('Job')
plt.ylabel('KPI')
plt.title('KPI1 for each Job - Machine A')
plt.xticks(range(1, len(kpis) + 1))
plt.grid(True)
plt.show()
