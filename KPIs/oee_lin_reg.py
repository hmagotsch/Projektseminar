
#Maschine A - KPI
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

    print(f'Job {job}: A = {A}, P = {P}, Q= {Q}, OEE = {OEE}, gross= {gained_gross}, speed = {Max_speed} , prod-time = {production_time.total_seconds()}')

# Create a line chart for KPI
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(OEEs) + 1), OEEs, color='green', label='OEE', marker='o')
plt.xlabel('Job')
plt.ylabel('OEE')
plt.title('Gesamtanlageneffektivität - Maschine A')
plt.xticks(range(1, len(OEEs) + 1))
plt.grid(True)
plt.show()

#Maschine A - Prediction
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import t
import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('MachineA_with_Job.csv', sep=';')

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
new_speed_value = 8000  # Replace with the new value you want to predict
predicted_value = slope * new_speed_value + intercept

# Get the standard error of the prediction
se = np.sqrt(model.scale)  # Standard error of the residuals

# Calculate the confidence interval for the prediction
alpha = 0.05  # Significance level
t_value = t.ppf(1 - alpha / 2, df=model.df_resid)  # T-distribution critical value
margin_of_error = t_value * se
prediction_ci = (predicted_value - margin_of_error, predicted_value + margin_of_error)

# Print the results
print("Predicted Single Variable:", predicted_value)
print("95% Confidence Interval:", prediction_ci[0], "to", prediction_ci[1])

# Create a scatter plot
plt.scatter(average_speed_per_job, single_variable, label='Data points')

# Plot the regression line
regression_line = model.predict(X)
plt.plot(average_speed_per_job, regression_line, color='red', label='Regression Line')

# Highlight the new value and its prediction with the correct confidence interval
plt.scatter(new_speed_value, predicted_value, color='green', label='Predicted Value')
plt.plot([new_speed_value, new_speed_value], [prediction_ci[0], prediction_ci[1]], color='green', linestyle='--', label='95% CI')

# Add labels and a title
plt.xlabel('Average Speed per Job')
plt.ylabel('OEE')
plt.title('Scatter Plot with Regression Line and Prediction Interval')

# Show the plot
plt.legend()
plt.show()
