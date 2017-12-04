import boto3
import table
from preprocessing import filter_array
import numpy as np
import datetime
import pickle

action_dict = {'going_down_stairs':0, 'staying_still':1, 'walking':2, 'going_up_stairs':3, 'running':4}

def save_cloud():
    device = '1'
    dynamodb = boto3.resource('dynamodb')
    # lines = table.scan_table("Location")['Items']
    # lines = table.scan_table_allpages("Location", 'device', 1)#['Items']
    lines = table.scan_table_allpages("Accelerometer", "Devicename", device)
    print(lines[0])

    data = [{"Timestamp": l['Timestamp'], 'Action': l['Action'],
             'X': float(l['X']), 'Y': float(l['Y']), 'Z': float(l['Z']), 'Actiontime': l['Actiontime']}
            for l in lines]

    data = sorted(data, key=lambda k: k['Timestamp'])
    dx = np.reshape([d['X'] for d in data], [-1, 1])
    dy = np.reshape([d['Y'] for d in data], [-1, 1])
    dz = np.reshape([d['Z'] for d in data], [-1, 1])
    daction = [d['Action'] for d in data]
    dtime = [d['Actiontime'] for d in data]

    def _convert_from_strf(time):
        return datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S').timestamp()
    def _convert_to_strf(time):
        return datetime.datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')

    timestamps = [_convert_from_strf(time) for time in dtime]
    real_time = [_convert_to_strf(timestamps[0])]
    start_val = timestamps[0]
    start_index = 0
    for i in range(1, len(dtime)):
        if timestamps[i] - timestamps[i-1] > 5:
            start_val = timestamps[i]
            start_index = i
        real_time.append(_convert_to_strf(start_val + np.floor((i-start_index)/50)))

    concated = np.concatenate([dx, dy, dz], axis=1)
    with open('acce_device_'+device, 'wb') as f:
        pickle.dump([concated, daction, real_time], f)

def load_cloud(D, Intervel, device='0'):
    with open('acce_device_'+device, 'rb') as f:
        concated, daction, real_time = pickle.load(f)
    x_test = filter_array(concated, D, Intervel)
    y_test = np.reshape([action_dict[daction[int(Intervel * i + D/2)]]
              for i in range(int(np.floor((concated.shape[0]-D) / float(Intervel))))], [-1])
    real_time = [real_time[int(Intervel * i + D/2)]
              for i in range(int(np.floor((concated.shape[0]-D) / float(Intervel))))]
    return x_test, y_test, real_time


if __name__ == '__main__':
    save_cloud()