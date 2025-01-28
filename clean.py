import ast
import pandas as pd
from statistics import mean as mean

data = pd.read_csv('data/borg_traces_data.csv')

# data = data.drop('time', axis = 1)
data = data.drop('user', axis = 1)
data = data.drop('Unnamed: 0', axis = 1)
data = data.drop('constraint', axis = 1)
data = data.drop('maximum_usage', axis = 1)
data = data.drop('collection_name', axis = 1)
data = data.drop('random_sample_usage', axis = 1)
data = data.drop('cycles_per_instruction',axis = 1)
data = data.drop('cpu_usage_distribution', axis = 1)
data = data.drop('collection_logical_name', axis = 1)
data = data.drop('start_after_collection_ids', axis = 1)
data = data.drop('tail_cpu_usage_distribution', axis = 1)
data = data.drop('memory_accesses_per_instruction',axis = 1)
data = data.dropna()

resource_request_cpu = []
resource_request_mem = []
for d in data['resource_request']:
    d = ast.literal_eval(d)
    resource_request_cpu.append(d['cpus'])
    resource_request_mem.append(d['memory'])

data['resource_request_cpus'] = resource_request_cpu
data['resource_request_memory'] = resource_request_mem
data = data.drop('resource_request', axis = 1)

average_usage_cpu = []
average_usage_mem = []
for d in data['average_usage']:
    d = ast.literal_eval(d)
    average_usage_cpu.append(d['cpus'])
    average_usage_mem.append(d['memory'])

data['average_usage_cpu'] = average_usage_cpu
data['average_usage_mem'] = average_usage_mem
data = data.drop('average_usage', axis = 1)

data.loc[data['event'] == 'LOST', 'failed'] = 1
data.loc[data['event'] == 'FAIL', 'failed'] = 1
data.loc[data['event'] == 'EVICT', 'failed'] = 1
data.loc[data['event'] == 'KILL', 'failed'] = 1
data = data.drop('event', axis = 1)

data.corr()

data = data.drop('instance_events_type', axis = 1)
data = data.drop('collections_events_type', axis = 1)
data = data.drop('sample_rate', axis = 1)
data = data.drop('average_usage_cpu', axis = 1)
data = data.drop('instance_index', axis = 1)
data = data.drop('machine_id', axis = 1)
data = data.drop('assigned_memory', axis = 1)
data = data.drop('average_usage_mem', axis = 1)
data = data.drop('alloc_collection_id', axis = 1)
data.to_csv('data/clean.csv', index = False)

print('done')