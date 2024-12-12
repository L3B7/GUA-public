import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy.fft import dct
import scipy.interpolate
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from gua_enums import *
import InteractiveLegend
import random
from tqdm import tqdm
from pywt import Wavelet
def CropAndPadArrayFront(arr, max_length,col, padding='zero'):
    if len(arr) > max_length:
        arr = arr[-max_length:]
        if(col=='timestamp'):
            return arr-arr[0]
        else:
            return arr
    elif len(arr) == max_length:
        return arr    
    else:
        pad_length = max_length - len(arr)
        if padding == 'linear':
            dt = np.mean(np.diff(arr))
            #print(dt)
            padded_arr = np.concatenate((np.linspace(0., arr[0]+dt*pad_length,num=pad_length, endpoint=False, retstep=False), arr+dt*pad_length))
            return padded_arr
        elif padding =='edge':
            padding_value = arr[0]
        elif padding =='zero':
            padding_value = np.zeros_like(arr[0],dtype=np.float32)
        else:
            padding_value = np.empty_like(arr[0])

        padded_arr = np.concatenate((np.repeat([padding_value], [pad_length], axis=0), arr))
        return padded_arr.astype(arr.dtype)

# arr= np.array([0,0.49,1,1.51,2])
# print(arr.shape)
# CropAndPadArrayFront(arr, 8,'linear')

def cropAndPadDfFront(df:pd.DataFrame, length = 4000, padding = ['linear','zero'], cols=['timestamp','sensordata']):
    assert len(np.asarray(padding)) == np.asarray(len(cols))
    for idx, col in enumerate(cols):
        print(col, ' : ',padding[idx])
        #print(df[col].shape)
        if(df[col].shape[0] != length):
            df[col] = df[col].apply(lambda x: CropAndPadArrayFront(x, length,col=col, padding= padding[idx]))
    return df
def cropFront(arr, max_length):
    if len(arr) > max_length:
        arr = arr[-max_length:]
        return arr
    else:
        return arr    
def rolling_window(a, window_size, step_size, mdebug=False)->list:
    if len(a)<window_size:
        return [CropAndPadArrayFront(a,window_size,"",padding='edge')]
    arr = []
    for i in range(len(a)-1, window_size-1, -step_size):
        if(mdebug):print(i-window_size,':',i)
        arr.append(a[i-window_size:i])
    arr.reverse()
    return arr

def plotOldAndNew(oldD, newD, mtitle="",num=5,startindex=0,yrange=3):
    plt.figure()
    plt.suptitle(mtitle)
    for i in range(startindex,startindex+num):
        plt.subplot(2,num,i-startindex+1)
        VALS = newD[i]
        plt.plot(VALS[:,0])
        plt.plot(VALS[:,1])
        plt.plot(VALS[:,2])
        plt.ylim(-yrange, yrange)
    for i in range(startindex,startindex+num):
        plt.subplot(2,num,num+i-startindex+1)
        VALS = oldD[i]
        plt.plot(VALS[:,0])
        plt.plot(VALS[:,1])
        plt.plot(VALS[:,2])
        plt.ylim(-yrange, yrange)
        #print(TS[-1])
    plt.subplot(2, num, 1)
    plt.title("Recreated data")
    plt.subplot(2, num, 6)
    plt.title("Original data")
    plt.gcf().set_size_inches(10,4)
    plt.tight_layout()
    plt.show() 

def plotPca(x_train,x_test, y_train, y_test, encoder, latent_dim, MLength, dn=3, al=0.5, title="PCA of autoencoder"):

    #colors:
    colors = ['black', 'red']  # train set
    unique_class_ids = np.unique(y_train)
    color_map = ListedColormap([colors[i % len(colors)] for i in range(len(unique_class_ids))])
    cs = color_map(y_train)
    colors2 = ['black', 'red' ]  # test set
    color_map2 = ListedColormap([colors2[i % len(colors2)] for i in range(len(unique_class_ids))])
    cs2 = color_map2(y_test)
    
    original_data = x_train
    encoded_data = encoder.encoder(original_data).numpy()
    decoded_data = encoder.decoder(encoded_data).numpy()

    original_test_data = x_test
    encoded_test_data = encoder.encoder(original_test_data).numpy()
    decoded_test_data = encoder.decoder(encoded_test_data).numpy()


# Create a PCA instance: pca
    pca_o = PCA(n_components=dn)
    pca_e = PCA(n_components=dn)
    pca_d = PCA(n_components=dn)

# Fit and transform the data array
    pca_original = pca_o.fit_transform(original_data.reshape(-1,MLength*3))
    pca_original_test = pca_o.transform(original_test_data.reshape(-1,MLength*3))

    pca_encoded = pca_e.fit_transform(encoded_data.reshape(-1,latent_dim))
    pca_encoded_test = pca_e.transform(encoded_test_data.reshape(-1,latent_dim))

    pca_decoded = pca_d.fit_transform(decoded_data.reshape(-1,MLength*3))
    pca_decoded_test = pca_d.transform(decoded_test_data.reshape(-1,MLength*3))

# Create a 3D plot
    if(dn==3):
        fig = plt.figure(figsize=(15,10))
        fig.suptitle(title)
        ax = fig.add_subplot(131, projection='3d')
        ax.scatter(pca_original[:,0], pca_original[:,1], pca_original[:,2], alpha=al, c=cs, marker='x', label='input')
        ax.scatter(pca_original_test[:,0], pca_original_test[:,1], pca_original_test[:,2], alpha=al, c=cs2, marker='o', label='input test')
        ax.legend(loc='upper right')
        ax = fig.add_subplot(132, projection='3d')
        ax.scatter(pca_encoded[:,0], pca_encoded[:,1], pca_encoded[:,2], alpha=al, c=cs, marker='x', label='encoded')
        ax.scatter(pca_encoded_test[:,0], pca_encoded_test[:,1], pca_encoded_test[:,2], alpha=al, c=cs2, marker='o', label='encoded test')
        ax.legend(loc='upper right')
        ax = fig.add_subplot(133, projection='3d')
        ax.scatter(pca_decoded[:,0], pca_decoded[:,1], pca_decoded[:,2], alpha=al, c=cs, marker='x', label='decoded')
        ax.scatter(pca_decoded_test[:,0], pca_decoded_test[:,1], pca_decoded_test[:,2], alpha=al, c=cs2, marker='o', label='decoded test')
        ax.legend(loc='upper right')
        plt.show()
    else:
        fig = plt.figure(figsize=(15,5))
        fig.suptitle(title)
        ax = fig.add_subplot(131)
        ax.scatter(pca_original[:,0], pca_original[:,1], alpha=al, c=cs, marker='x', label='input')
        ax.scatter(pca_original_test[:,0], pca_original_test[:,1], alpha=al, c=cs2, marker='o', label='input test')
        ax.legend(loc='upper right')
        ax = fig.add_subplot(132)
        ax.scatter(pca_encoded[:,0], pca_encoded[:,1], alpha=al, c=cs, marker='x', label='encoded')
        ax.scatter(pca_encoded_test[:,0], pca_encoded_test[:,1], alpha=al, c=cs2, marker='o', label='encoded test')
        ax.legend(loc='upper right')
        ax = fig.add_subplot(133)
        ax.scatter(pca_decoded[:,0], pca_decoded[:,1], alpha=al, c=cs, marker='x', label='decoded')
        ax.scatter(pca_decoded_test[:,0], pca_decoded_test[:,1], alpha=al, c=cs2, marker='o', label='decoded test')
        ax.legend(loc='upper right')
        plt.show()
        
        
def find_last_plateau(data,ts, diff_threshold=0.03, std_treshold=0.01, L=20,skipback=300)->int:
    """
    Find the last plateau of length L in the given magnitude array, using the given threshold.
    Default takes magnitude data, accounted for gravity.
    Returns the index of the end of the plateau, or None if no plateau is found.
    """
    plateau_end = None
    plateau_length = 0
    for i in range(len(data) -1-skipback, 1, -1):
        if abs(data[i-1] - data[i]) < diff_threshold:
            if plateau_end is None:
                plateau_end = i
                plateau_length = 1
            else:
                if np.std(data[plateau_end-plateau_length-1:plateau_end]) > std_treshold:
                    plateau_end = None
                    plateau_length = 0
                    continue
                    
                plateau_length += 1
            if plateau_length == L:
                break
        else:
            plateau_end = None
            plateau_length = 0

    if plateau_length == L:
        return plateau_end # type: ignore
    else:
        return 0

SCALERATE = 10
pL = lambda x: min(max(np.ceil(len(x)/10),50),100)
diff = lambda x: np.append(np.sum(abs(x[1:len(x),:] - x[0:-1,:]), axis=1) * SCALERATE,0)
flp = lambda x,y:find_last_plateau(diff(x),y, L=pL(x),diff_threshold=0.03, std_treshold=0.01)
def process_record(record):

    flpIndex = flp(record['sensordata.GRV'], record['timestamp.GRV'])
    startTs = record['timestamp.GRV'][flpIndex]
    record['start'] = startTs 
    
    for stype in SensorEnum.valid():
        stypeStartIndex = np.searchsorted(record['timestamp.'+stype.name], startTs)
        if (stypeStartIndex ==record['timestamp.'+stype.name].size):
            stypeStartIndex = None;
        record['length.' +stype.name] = (record['timestamp.'+stype.name][-1] - record['timestamp.'+stype.name][0]) *10**(-9)
        record['startindex.'+stype.name] = stypeStartIndex
    return record


def interpolate_sensor_data(record, runup, length_sec, num_samples):
    """Interpolate the sensor data to have the same length for all sensors"""
    flpIndex = flp(record['sensordata.GRV'], record['timestamp.GRV'])
    startTs = max(record['timestamp.GRV'][flpIndex] - int(runup * 10**9),0)
    endTs = startTs + int(length_sec * 10**9)
    for stype in SensorEnum.valid():
        mask = (record['timestamp.'+stype.name] >= startTs) & (record['timestamp.'+stype.name] <= endTs)
        record['sensordata.'+stype.name] = record['sensordata.'+stype.name][mask]
        record['timestamp.'+stype.name] = record['timestamp.'+stype.name][mask]      

        if len(record['timestamp.'+stype.name]) == 0:
            record['timestamp.'+stype.name] = None
            continue

        interp_t = np.linspace(startTs, endTs, num=num_samples)

        interp_func_t = scipy.interpolate.interp1d(record['timestamp.'+stype.name], record['timestamp.'+stype.name], kind='linear', fill_value='extrapolate')
        interp_func_v = scipy.interpolate.interp1d(record['timestamp.'+stype.name], record['sensordata.'+stype.name], axis=0, kind='linear', fill_value='extrapolate')

        record['timestamp.'+stype.name] = interp_func_t(interp_t)
        record['sensordata.'+stype.name] = interp_func_v(interp_t)
    return record

def strip_front(record, runup:float=0):
    print(record)
    flpIndex = flp(record['sensordata.GRV'], record['timestamp.GRV'])
    startTs = max(record['timestamp.GRV'][flpIndex] - int(runup * 10**9),0)
    #record['startts'] = startTs
    for stype in SensorEnum.valid():
        #print(record.name)
        #print(stype.name, startTs, record['timestamp.'+stype.name].size)
        stypeStartIndex = np.searchsorted(record['timestamp.'+stype.name], startTs)
        
        
        #print(stypeStartIndex)
        if (stypeStartIndex !=record['timestamp.'+stype.name].size):
            record['sensordata.'+stype.name] = record['sensordata.'+stype.name][stypeStartIndex:]
            record['timestamp.'+stype.name] = record['timestamp.'+stype.name][stypeStartIndex:]
        else:
            stypeStartIndex = None
            record['sensordata.'+stype.name] = None    
            record['timestamp.'+stype.name] = None
    return record


def pad_end(record, slist, desired_length=820):

    for stype in slist:
        #print(desired_length,len(record['sensordata.'+stype.name]),desired_length-len(record['sensordata.'+stype.name]))
        if len(record['sensordata.'+stype.name]) < desired_length:
            record['sensordata.'+stype.name] = np.append(record['sensordata.'+stype.name],np.zeros((desired_length-len(record['sensordata.'+stype.name]),SensorEnum.dim(stype)),dtype=np.float32),axis=0)
        else:
            record['sensordata.'+stype.name] = record['sensordata.'+stype.name][:desired_length]
        #print(desired_length,len(record['sensordata.'+stype.name]),desired_length-len(record['sensordata.'+stype.name]))
    return record 

def unit_vector(record):
    for stype in SensorEnum.valid():
        dim = SensorEnum.dim(stype)
        norm = np.tile(np.linalg.norm(record['sensordata.'+stype.name],axis=1),(dim,1)).T
        record['sensordata.'+stype.name] = record['sensordata.'+stype.name]/norm
    return record

def dim_normalize(data_set):
    for i in range(data_set.shape[0]):
        dim_normalize_data(data_set[i,:,:])
    return data_set

def dim_normalize_data(data):
    """Normalize the data along x,y,z axis. Independent on each sample"""
    for i in range(data.shape[1]):
        mean=np.mean(data[:,i],axis=0)
        std=np.std(data[:,i],axis=0)
        data[:,i]=(data[:,i]-mean)/std
    return data

def random_shift(all_x_train,all_y_train,num,shift_range):
    ag_x = []
    ag_y = []
    for i in range(len(all_x_train)):
        ag_x.append(all_x_train[i])
        ag_y.append(all_y_train[i])
        smp = random.sample(range(1,shift_range*2+1),num)
        for j in range(num):
            ag_x.append(np.roll(all_x_train[i],smp[j],axis=0))
            ag_y.append(all_y_train[i])
    return np.array(ag_x),np.array(ag_y)


def moving_average(row, sensor_names,window_size = 14):
    new_row = row.copy()
    for sensor_name in sensor_names:
        num_features = row["sensordata." +sensor_name].shape[1]
        convolved_data = np.zeros((row["sensordata." +sensor_name].shape[0] - window_size + 1, num_features))
        #print(f'convolved_data shape: {convolved_data.shape}')
        for i in range(num_features):
            convolved_data[:, i] = moving_avg(row["sensordata." +sensor_name][:, i],window_size)

        new_row["sensordata." +sensor_name] = convolved_data
        start = int((window_size - 1) / 2)
        end = -start-1
        new_row['timestamp.' +sensor_name] = row['timestamp.' +sensor_name][start:end]
    return new_row

def moving_avg(row, window_size, weights=None):
    if weights is None:
        weights = np.ones(window_size) / window_size
    return np.convolve(row, weights, mode='valid')
def apply_wavelet_denoising_row(row, sensor_names,threshold=0.5, wavelet_s='coif5', level=2):
    new_row = row.copy()
    for sensor_name in sensor_names:
        num_features = row["sensordata." + sensor_name].shape[1]
        denoised_data = np.zeros_like(row["sensordata." + sensor_name])
        for i in range(num_features):
            wavelet = Wavelet(wavelet_s)
            WaveletCoeffs = pywt.wavedec(row["sensordata." + sensor_name][:, i], wavelet, level=level)
            NewWaveletCoeffs = map (lambda x: pywt.threshold(x, value=threshold, mode="soft"),WaveletCoeffs)
            denoised_feature = pywt.waverec(list(NewWaveletCoeffs), wavelet)

            # Ensure denoised data has the same length as the original data
            if len(denoised_feature) > len(row["sensordata." + sensor_name][:, i]):
                denoised_feature = denoised_feature[:len(row["sensordata." + sensor_name][:, i])]
            elif len(denoised_feature) < len(row["sensordata." + sensor_name][:, i]):
                denoised_feature = np.concatenate([denoised_feature, np.zeros(len(row["sensordata." + sensor_name][:, i]) - len(denoised_feature))])
            denoised_data[:, i] = denoised_feature
        new_row["sensordata." + sensor_name] = denoised_data
    return new_row

def apply_wavelet_denoising(window, threshold=0.9, wavelet_s='coif5', level=2):
    if len(window.shape) == 1:
        num_features = 1
    else:
        num_features = window.shape[1]
    denoised_data = np.zeros_like(window)
    for i in range(num_features):
        denoised_feature = _apply_wavelet_denoising_feature(window[:, i], threshold, wavelet_s, level)
        denoised_data[:, i] = denoised_feature
    return denoised_data
def _apply_wavelet_denoising_feature(window, threshold=0.9, wavelet_s='coif5', level=2):
    wavelet = Wavelet(wavelet_s)
    wavelet = Wavelet(wavelet_s)
    WaveletCoeffs = pywt.wavedec(window, wavelet, level=level)
    NewWaveletCoeffs = map (lambda x: pywt.threshold(x, value=threshold, mode="soft"), WaveletCoeffs)
    denoised_feature = pywt.waverec(list(NewWaveletCoeffs), wavelet)
    if len(denoised_feature) > len(window):
        denoised_feature = denoised_feature[:len(window)]
    elif len(denoised_feature) < len(window):
        denoised_feature = np.concatenate([denoised_feature, np.zeros(len(window) - len(denoised_feature))])
    return denoised_feature

def create_windows(data, window_size, overlap):
    window_shape = [window_size]
    window_shape.extend(list(data.shape[1:]))
    sliding = np.lib.stride_tricks.sliding_window_view(data, window_shape=window_shape)[::(window_size-int(window_size*overlap))]
    return sliding


"""
Creates a list of timestamps for windowing accross sensors
"""
def create_window_ts(sensor_timestamps:list[np.ndarray],window_duration_sec:float, overlap:float = 0.5, mode:str = 'narrow', start_ts = 0)->list:
    window_duration = int(window_duration_sec * 1000000000)
    ts = []
    mins = []
    mins.append(start_ts)
    maxs = []
    for sensor in sensor_timestamps:
        mins.append(np.min(sensor))
        maxs.append(np.max(sensor))
    if mode == 'narrow':
        start = np.max(mins)
        end = np.min(maxs)
    elif mode == 'wide':
        start = np.min(mins)
        end = np.max(maxs)
        
    #print([minn-min(mins) for minn in mins])
    for timer in range(start,end,int(window_duration - (overlap*window_duration))):
        ts.append(timer)
    if(len(ts) != 0):
        ts.pop()
    ts.append(end-window_duration)
    ts.append(end)
    return [window_duration, ts]

"""
Creates the windowed version of the sensor_data based on the timestamps
"""
def create_windows_for_ts(data,timestamp,ts:list,window_duration:int)->list:
    assert len(data) == len(timestamp)
    windows = []
    #print(data.shape)
    for starts in ts[:-1]:
        window = data[(timestamp >= starts) & (timestamp < starts+window_duration)]
        #print(window.shape)
        windows.append(window)
    return windows
"""
Creates a list of windows for each sensor from a df row, where the windows are synced based on the timestamps.
Uses the GRV sensor to determine the start of the window with find_last_plateau.
Can be converted to a DataFrame.
pd.DataFrame(columns=['uid', 'window_index']+sensor_columns)
"""
def create_synced_windows(row,window_duration_sec:float, overlap:float = 0.5, interpolate:bool = False, runup:float=0)->list:
    start_idx = flp(row['sensordata.GRV'], row['timestamp.GRV'])
    start_ts = row['timestamp.GRV'][start_idx] - int(runup * 1000000000)
    
    data_dict = {sensor.name: [] for sensor in SensorEnum.valid()}
    [duration, tss] = create_window_ts([row[sensor.timestampColumn()] for sensor in SensorEnum.valid()], window_duration_sec, overlap, 'narrow', start_ts)
    for sensor in SensorEnum.valid():
        windows = create_windows_for_ts(row[sensor.dataColumn()],row[sensor.timestampColumn()],tss,duration)
        window_num = len(windows)
        data_dict[sensor.name] = windows
    window_results =[]
    for i in range(window_num):
        dct = {}
        for sensor in SensorEnum.valid():
            if data_dict[sensor.name][i].shape[0] != 0:
                dct[sensor.name] = data_dict[sensor.name][i]
                continue
            dct[sensor.name] = None
        window_results.append(dct)
        #print(data_dict[sensor.name][i].shape)
    return window_results
            

"""
def split_features_and_adjust_labels(x, y, window_size=100, overlap=0.4):
    num_samples = x.shape[0]
    num_timesteps = x.shape[1]
    num_features = x.shape[2]
    
    step_size = int(window_size * (1 - overlap))
    num_segments = int((num_timesteps - window_size) / step_size) + 1

    split_x = np.zeros((num_samples * num_segments, window_size, num_features))
    adjusted_y = np.repeat(y, num_segments)

    segment_index = 0
    for sample_index in range(num_samples):
        for i in range(0, num_timesteps - window_size + 1, step_size):
          split_x[segment_index] = x[sample_index, i : i + window_size, :]
          segment_index += 1
          
    return split_x, adjusted_y
    """

def create_windows_interpol(data, window_size:int = 100, overlap:float = 0.2)->list:
    step_size = int(window_size * (1 - overlap))
    n_windows = ((data.shape[0] - window_size) // step_size) + 1
    windows = [None] * n_windows
    # Create windows
    for i in range(n_windows):
        start_idx = i * step_size
        windows[i] = data[start_idx:start_idx + window_size]
    return windows


""" assert len(data) == len(timestamp)
    windows = []
    #print(data.shape)
    for starts in ts[:-1]:
        window = data[(timestamp >= starts) & (timestamp < starts+window_duration)]
        #print(window.shape)
        windows.append(window)
    return windows"""

def create_synced_windows_interpol(row,window_size:int, overlap:float = 0.5)->list:    
    data_dict = {sensor.name: [] for sensor in SensorEnum.valid()}

    for sensor in SensorEnum.valid():
        windows = create_windows_interpol(row[sensor.dataColumn()],window_size,overlap)
        window_num = len(windows)
        data_dict[sensor.name] = windows
    window_results =[]
    for i in range(window_num):
        dct = {}
        for sensor in SensorEnum.valid():
            if data_dict[sensor.name][i].shape[0] != 0:
                dct[sensor.name] = data_dict[sensor.name][i]
                continue
            dct[sensor.name] = None
        window_results.append(dct)
        #print(data_dict[sensor.name][i].shape)
    return window_results

def get_columns_for_section(data_features:pd.DataFrame,i:int)->list:
    result=[]
    for sensor in SensorEnum.valid():
        result.append(np.stack(data_features[f"{sensor.name}{i}"].values)) # type: ignore
    return result

def get_features_and_labels(df, myfun, num_sections=3,selected_sections=[0,1,2], label_encoder:LabelEncoder|None=None):
    sensor_columns = [f"{sensor.name}" for sensor in SensorEnum.valid()]
    x = create_features_sectioned_pipe(df[sensor_columns],myfun, num_sections=num_sections, selected_sections=selected_sections)
    y_:pd.Series = df['uid']
    print(y_.value_counts())
    if label_encoder is not None:
        y=label_encoder.transform(y_)
    return x, np.asarray(y)

def create_features_sectioned_pipe(x, myfun, num_sections=3,selected_sections=[0,1,2] )->np.ndarray:
    out_data = None
    outrow = None
    for i in range(0,x.shape[0]):
        #print(i)
        save_sections = []
        for sensor in SensorEnum.valid():
            sensor_data = x[sensor.name].values[i]
            if(num_sections == 1):
                save_sections.append(myfun(sensor_data))
            else:
                section_splits = [int(len(sensor_data)/num_sections * i) for i in range(1,num_sections)]
                sections_prop = np.split(sensor_data, section_splits)
                sections = []
                for j in selected_sections:
                    sections.append(myfun(sections_prop[j])) #todo
                new_row = np.vstack(sections)            
                save_sections.append(new_row)
        combined = np.hstack(save_sections)
        outrow=combined.flatten()
        if out_data is None:
            out_data = outrow
        else:
            out_data = np.vstack([out_data,outrow])
    #print(out_data.shape) # type: ignore
    return out_data # type: ignore