import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
import pandas as pd
import matplotlib.pyplot as plt
from pydub import AudioSegment
import re
path = "D:/Downloads/UCL_data/wav"


def split_wav_files_in_directory(directory_path, chunk_length_ms):  # chunk_length_ms is 4 minutes in milliseconds
    # Create a 'splitted' subdirectory if it doesn't exist
    os.makedirs(os.path.join(directory_path, 'splitted'), exist_ok=True)
    c = os.listdir(directory_path)
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            audio = AudioSegment.from_wav(file_path)
            
            # Only process files longer than 4 minutes
            if len(audio) > chunk_length_ms:
                split_and_save_wav(file_path, directory_path, filename, chunk_length_ms)

def split_and_save_wav(file_path, directory_path, filename, chunk_length_ms):
    audio = AudioSegment.from_wav(file_path)
    chunks = make_chunks(audio, chunk_length_ms)
    
    for i, chunk in enumerate(chunks):
        chunk_name = f"{filename[:-4]}_part{i}.wav"
        chunk_path = os.path.join(directory_path, 'splitted', chunk_name)
        print(f"Exporting {chunk_path}...")
        chunk.export(chunk_path, format="wav")

def make_chunks(audio_segment, chunk_length_ms):
    number_of_chunks = len(audio_segment) // chunk_length_ms
    return [audio_segment[i * chunk_length_ms:(i + 1) * chunk_length_ms] for i in range(number_of_chunks)]


#split_wav_files_in_directory(path,chunk_length_ms=round(minutes*60000))


# Generate the prefix to filter the files
data_to_save = []




def plot_average_trial_time(path):
    # Initialize a dictionary to store the average duration for each trial
    trial_averages = {}

    # Iterate over trial numbers
    for trial_number in range(0, 12):
        durations = []
        
        # Iterate over speaker numbers
        for speaker_number in range(55):
            prefix = f'speaker{speaker_number}_channel1_trial{trial_number}'
            
            # Find the WAV file with the given prefix
            wav_files = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith('.wav')]
            
            # Process each WAV file
            for wav_file in wav_files:
                # Read the WAV file
                sample_rate, data = wavfile.read(os.path.join(path, wav_file))
                duration = len(data) / float(sample_rate)
                durations.append(duration)
        
        # Calculate the average duration for the trial
        if(len(durations) > 0):
            trial_averages[trial_number] = sum(durations) / len(durations)

    # Plotting the averages
    plt.figure(figsize=(15, 5))
    plt.bar(trial_averages.keys(), trial_averages.values())
    plt.xlabel('Trial Number')
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Time for Each Trial')
    plt.xticks(range(1, 12))
    plt.tight_layout()
    plt.show()

    return trial_averages
def plot_average_participant_time(path):
    # Initialize a dictionary to store the average duration for each speaker
    participant_averages = {}

    # Iterate over speaker numbers
    for speaker_number in range(1,55):
        durations = []
        
        # Iterate over trial numbers
        for trial_number in range(1, 12):
            prefix = f'speaker{speaker_number}_channel1_trial{trial_number}'
            
            # Find the WAV file with the given prefix
            wav_files = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith('.wav')]
            
            # Process each WAV file
            for wav_file in wav_files:
                # Read the WAV file
                sample_rate, data = wavfile.read(os.path.join(path, wav_file))
                duration = len(data) / float(sample_rate)
                durations.append(duration)
        
        # Calculate the average duration for the speaker
        participant_averages[speaker_number] = sum(durations) / len(durations)

    # Plotting the averages
    plt.figure(figsize=(15, 5))
    plt.bar(participant_averages.keys(), participant_averages.values())
    plt.xlabel('Speaker Number')
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Time for Each Participant')
    plt.xticks(range(1,55))
    plt.tight_layout()
    plt.show()

    return participant_averages



def split_wav(path):

    # List all files in the directory and filter out the WAV files with the given prefix and trial number
    for speaker_number in range(55):
        for trail_number in range(1,11):
            prefix = f'speaker{str(speaker_number)}'
            c = f'trial{str(trail_number)}'
            print(c)

            # Define the regex pattern
            pattern = re.compile(f"^{re.escape(prefix)}.*{re.escape(c)}.*")

            # List files matching the pattern
            len_wav_files = [f for f in os.listdir(path) if pattern.match(f)]
            for part in range(int(len(len_wav_files)/3)):
                prefix = f'speaker{str(speaker_number)}'
                c = f'trial{str(trail_number)}_part{part}'
                print(c)

                # Define the regex pattern
                pattern = re.compile(f"^{re.escape(prefix)}.*{re.escape(c)}.*")
                wav_files = [f for f in os.listdir(path) if pattern.match(f)]

                # Check if we have 3 files for the channels
                if len(wav_files) != 3:
                    print("Expected 3 files, but found:", len(wav_files))
                else:
                # plt.figure(figsize=(10, 7))  # Set the figure size
                    upper_belt_data = []
                    lower_belt_data = []
                    times_data = []
                    wav_file = ""
                    for i, wav_file in enumerate(sorted(wav_files)):
                        # Read the WAV file
                        sample_rate, data = wavfile.read(os.path.join(path, wav_file))

                        # Resample data to 25 Hz
                        target_sample_rate = 25
                        num_samples = round(len(data) * target_sample_rate / sample_rate)
                        resampled_data = resample(data, num_samples)

                        # Normalize the data using Z-score normalization
                        min_val = np.min(resampled_data)
                        max_val = np.max(resampled_data)

                        # Normalize the data to the range [0, 1]
                        normalized_data_0_1 = (resampled_data - min_val) / (max_val - min_val)

                        # Scale to the range [-1, 1]
                        scaled_data = (normalized_data_0_1 * 2) - 1
                        normalized_data =np.round(scaled_data, 5)

                        # Create a time array in seconds
                        times = np.arange(len(normalized_data)) / float(target_sample_rate)
                        if i == 0:
                            times_data = times.tolist()
                        elif i == 1:
                            upper_belt_data = normalized_data

                        elif i == 2:
                            lower_belt_data = normalized_data
                        else:
                            print("Too many channels in data; will skip data after the 3rd channel!")
                        #plt.subplot(3, 1, i+1)
                        #plt.plot(times, normalized_data)
                        #plt.title(f'Z-score Normalized Plot of {wav_file}')
                        #plt.ylabel('Normalized Amplitude')
                        #plt.xlabel('Time (s)')
                    for i, time in enumerate(times_data):
                        if upper_belt_data[i] < -10:
                            print(upper_belt_data[i])
                        data_to_save.append([wav_file, time, upper_belt_data[i]])
                    ##Create a subplot for each WAV file

                    #plt.tight_layout()
                    #plt.show()
    # Convert the list to a DataFrame
    return data_to_save

def create_output(path, output_csv='output.csv'):
    data_to_save = []

    # List all files in the directory and filter out the WAV files with the given prefix and trial number
    for speaker_number in range(55):
        for trial_number in range(1, 11):
            prefix = f'speaker{str(speaker_number)}'
            c = f'trial{str(trial_number)}'
            print(c)

            # Define the regex pattern
            pattern = re.compile(f"^{re.escape(prefix)}.*{re.escape(c)}.*")

            # List files matching the pattern
            len_wav_files = [f for f in os.listdir(path) if pattern.match(f)]
            for part in range(int(len(len_wav_files) / 3)):
                prefix = f'speaker{str(speaker_number)}'
                c = f'trial{str(trial_number)}_part{part}'
                print(c)

                # Define the regex pattern
                pattern = re.compile(f"^{re.escape(prefix)}.*{re.escape(c)}.*")
                wav_files = [f for f in os.listdir(path) if pattern.match(f)]

                # Check if we have 3 files for the channels
                if len(wav_files) != 3:
                    print("Expected 3 files, but found:", len(wav_files))
                else:
                    upper_belt_data = []
                    times_data = []
                    wav_file = ""
                    for i, wav_file in enumerate(sorted(wav_files)):
                        # Read the WAV file
                        sample_rate, data = wavfile.read(os.path.join(path, wav_file))

                        # Check if the audio is 4 minutes or longer
                        duration = len(data) / sample_rate
                        if duration < 4 * 60:
                            continue  # Skip files shorter than 4 minutes

                        # Cut the audio to 4 minutes if it's longer
                        if duration > 4 * 60:
                            data = data[:4 * 60 * sample_rate]

                        # Resample data to 25 Hz
                        target_sample_rate = 25
                        num_samples = round(len(data) * target_sample_rate / sample_rate)
                        resampled_data = resample(data, num_samples)

                        # Normalize the data using Z-score normalization
                        min_val = np.min(resampled_data)
                        max_val = np.max(resampled_data)

                        # Normalize the data to the range [0, 1]
                        normalized_data_0_1 = (resampled_data - min_val) / (max_val - min_val)

                        # Scale to the range [-1, 1]
                        scaled_data = (normalized_data_0_1 * 2) - 1
                        normalized_data = np.round(scaled_data, 5)

                        # Create a time array in seconds
                        times = np.arange(len(normalized_data)) / float(target_sample_rate)
                        if i == 0:
                            times_data = times.tolist()
                        elif i == 1:
                            upper_belt_data = normalized_data

                    for i, time in enumerate(times_data):
                        data_to_save.append([wav_file, time, upper_belt_data[i]])
                    

    # Convert the list to a DataFrame
    df = pd.DataFrame(data_to_save, columns=['filename', 'timeFrame', 'upper_belt'])

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    return df
import os
import re
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.io import wavfile

import os
import re
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.io import wavfile

def create_output(path, output_csv='output.csv'):
    data_to_save = []
    all_upper_belt_data = []

    # List all files in the directory and filter out the WAV files with the given prefix and trial number
    for speaker_number in range(55):
        for trial_number in range(1, 11):
            prefix = f'speaker{str(speaker_number)}'
            c = f'trial{str(trial_number)}'
            print(c)

            # Define the regex pattern
            pattern = re.compile(f"^{re.escape(prefix)}.*{re.escape(c)}.*")

            # List files matching the pattern
            len_wav_files = [f for f in os.listdir(path) if pattern.match(f)]
            for part in range(int(len(len_wav_files) / 3)):
                prefix = f'speaker{str(speaker_number)}'
                c = f'trial{str(trial_number)}_part{part}'
                print(c)

                # Define the regex pattern
                pattern = re.compile(f"^{re.escape(prefix)}.*{re.escape(c)}.*")
                wav_files = [f for f in os.listdir(path) if pattern.match(f)]

                # Check if we have 3 files for the channels
                if len(wav_files) != 3:
                    print("Expected 3 files, but found:", len(wav_files))
                else:
                    for i, wav_file in enumerate(sorted(wav_files)):
                        # Read the WAV file
                        sample_rate, data = wavfile.read(os.path.join(path, wav_file))

                        # Check if the audio is 4 minutes or longer
                        duration = len(data) / sample_rate
                        if duration < 4 * 60:
                            continue  # Skip files shorter than 4 minutes

                        # Cut the audio to 4 minutes if it's longer
                        if duration > 4 * 60:
                            data = data[:4 * 60 * sample_rate]

                        # Resample data to 25 Hz
                        target_sample_rate = 25
                        num_samples = round(len(data) * target_sample_rate / sample_rate)
                        resampled_data = resample(data, num_samples)

                        # Collect all upper belt data for normalization
                        if i == 1:
                            all_upper_belt_data.extend(resampled_data)

    # Calculate global min and max for normalization
    global_min = np.min(all_upper_belt_data)
    global_max = np.max(all_upper_belt_data)

    # Normalize and save data
    for speaker_number in range(55):
        for trial_number in range(1, 11):
            prefix = f'speaker{str(speaker_number)}'
            c = f'trial{str(trial_number)}'
            print(c)

            # Define the regex pattern
            pattern = re.compile(f"^{re.escape(prefix)}.*{re.escape(c)}.*")

            # List files matching the pattern
            len_wav_files = [f for f in os.listdir(path) if pattern.match(f)]
            for part in range(int(len(len_wav_files) / 3)):
                prefix = f'speaker{str(speaker_number)}'
                c = f'trial{str(trial_number)}_part{part}'
                print(c)

                # Define the regex pattern
                pattern = re.compile(f"^{re.escape(prefix)}.*{re.escape(c)}.*")
                wav_files = [f for f in os.listdir(path) if pattern.match(f)]

                # Check if we have 3 files for the channels
                if len(wav_files) != 3:
                    print("Expected 3 files, but found:", len(wav_files))
                else:
                    upper_belt_data = []
                    times_data = []
                    wav_file = ""
                    for i, wav_file in enumerate(sorted(wav_files)):
                        # Read the WAV file
                        sample_rate, data = wavfile.read(os.path.join(path, wav_file))

                        # Check if the audio is 4 minutes or longer
                        duration = len(data) / sample_rate
                        if duration < 4 * 60:
                            continue  # Skip files shorter than 4 minutes

                        # Cut the audio to 4 minutes if it's longer
                        if duration > 4 * 60:
                            data = data[:4 * 60 * sample_rate]

                        # Resample data to 25 Hz
                        target_sample_rate = 25
                        num_samples = round(len(data) * target_sample_rate / sample_rate)
                        resampled_data = resample(data, num_samples)

                        # Normalize the data to the range [-1, 1]
                        normalized_data = 2 * (resampled_data - global_min) / (global_max - global_min) - 1

                        # Create a time array in seconds
                        times = np.arange(len(normalized_data)) / float(target_sample_rate)
                        if i == 0:
                            times_data = times.tolist()
                        elif i == 1:
                            upper_belt_data = normalized_data

                    for i, time in enumerate(times_data):
                        data_to_save.append([wav_file, time, upper_belt_data[i]])

    # Convert the list to a DataFrame
    df = pd.DataFrame(data_to_save, columns=['filename', 'timeFrame', 'upper_belt'])

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

    return df




# Example usage
# create_output('path_to_wav_files')


# Example usage
# Example usage:
# Specify the directory containing your WAV files
import pandas as pd
import numpy as np

def normalize_upper_belt(df, output_csv='normalized_output.csv'):
    # Read the CSV file into a DataFrame

    # Get unique filenames
    unique_filenames = df['filename'].unique()

    # Convert the upper_belt column to strings
    df['upper_belt'] = df['upper_belt'].astype(str)

    # Get unique filenames
    unique_filenames = df['filename'].unique()

    # Initialize a list to store normalized data
    normalized_data = []

    # Normalize the upper_belt data for each filename
    for filename in unique_filenames:
        # Filter data for the current filename
        file_data = df[df['filename'].str.contains(filename, case=False, na=False)]

        # Check for "?" in the upper_belt data and skip those rows
        if file_data['upper_belt'].str.contains('\?').any():
            continue

        # Get the upper_belt data and convert to float
        upper_belt_data = file_data['upper_belt'].astype(float).values

        # Normalize the data to the range [-1, 1]
        min_val = np.min(upper_belt_data)
        max_val = np.max(upper_belt_data)
        normalized_upper_belt = 2 * (upper_belt_data - min_val) / (max_val - min_val) - 1

        # Create a DataFrame for the normalized data
        normalized_file_data = pd.DataFrame({
            'filename': filename,
            'upper_belt': normalized_upper_belt
        })

        # Append the normalized data to the list
        normalized_data.append(normalized_file_data)

    # Concatenate all normalized data into a single DataFrame
    normalized_df = pd.concat(normalized_data, ignore_index=True)

    # Save the normalized DataFrame to a new CSV file
    normalized_df.to_csv(output_csv, index=False)

    return normalized_df

# Example usage
# normalized_df = normalize_upper_belt('input.csv')

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def match_upperbelt_data(gen_labels_path, og_labels_path, threshold=0.5):
    # Load the CSV files into DataFrames
    gen_labels = pd.read_csv(gen_labels_path)
    og_labels = pd.read_csv(og_labels_path)
    #gen_labels =normalize_upper_belt(gen_labels)
    #og_labels =normalize_upper_belt(og_labels)

    # Initialize a list to store the results
    matches = []
    unique_gen = gen_labels['filename'].unique()
    unique_og = og_labels['filename'].unique()
    print(len(unique_gen))
    # Iterate over each unique filename in the generated labels
    counter = 0
    for gen_filename in unique_gen:
        gen_data = gen_labels[gen_labels['filename'].str.contains(gen_filename, case=False, na=False)].values
        print(counter)
        counter+=1
        # Iterate over each unique filename in the original labels
        for og_filename in unique_og:
            og_data = og_labels[og_labels['filename'].str.contains(og_filename, case=False, na=False)].values

            # Extract the upper_belt data
            d1 = og_data[:, -1]
            d2 = gen_data[:, -1]

            # Filter out entries containing '?'
            valid_indices = (d1 != '?') & (d2 != '?')
            d1 = d1[valid_indices].astype(float)
            d2 = d2[valid_indices].astype(float)


            # Calculate the Pearson correlation coefficient
            if len(d1) > 0 and len(d2) > 0:  # Ensure there are valid entries to compare
                correlation, _ = pearsonr(d1, d2)
                # If the correlation is above the threshold, add it to the results
                if correlation >= threshold:
                    print(correlation)
                    matches.append((gen_filename, og_filename, correlation * 100))

    return matches



import librosa




def read_wav(file_path, sr):

    data, rate = librosa.load(file_path, sr=sr)

    return rate, data



def z_normalize(data):

    return (data - np.mean(data)) / np.std(data)



def trim_to_4_minutes(rate, data):

    max_length = 4 * 60 * rate  # 4 minutes in samples

    if len(data) > max_length:

        return data[:max_length]

    return data



def resample_to_1000(data, original_rate):

    target_rate = 1000

    data_resampled = librosa.resample(data, orig_sr=original_rate, target_sr=target_rate)

    return data_resampled




def calculate_similarity(data1, data2):
    # Compute MFCC features
    mfcc1 = librosa.feature.mfcc(y=data1, sr=1000)
    mfcc2 = librosa.feature.mfcc(y=data2, sr=1000)

    # Flatten the MFCC arrays
    mfcc1_flat = mfcc1.flatten()
    mfcc2_flat = mfcc2.flatten()

    # Compute weights based on the magnitude of the values
    weights = np.abs(mfcc1_flat) + np.abs(mfcc2_flat)

    # Compute weighted correlation
    weighted_corr = np.corrcoef(mfcc1_flat * weights, mfcc2_flat * weights)[0, 1]

    return weighted_corr


def compare_wav_files(folder1, folder2, threshold=0.5):

    results = []

    for file1 in os.listdir(folder1):

        if file1.endswith('.wav'):

            rate1, data1 = read_wav(os.path.join(folder1, file1),1000)

            if len(data1) < 4 * 60 * rate1:

                continue

            data1 = trim_to_4_minutes(rate1, data1)

            data1 = resample_to_1000(data1, rate1)

            data1 = z_normalize(data1)

            highest_similarity = 0

            most_similar_file2 = None

            for file2 in os.listdir(folder2):

                if file2.endswith('.wav') and 'channel1' in file2:

                    rate2, data2 = read_wav(os.path.join(folder2, file2),1000)

                    if len(data2) < 4 * 60 * rate2:

                        continue

                    data2 = trim_to_4_minutes(rate2, data2)

                    data2 = resample_to_1000(data2, rate2)

                    data2 = z_normalize(data2)

                    similarity = calculate_similarity(data1, data2)

                    if similarity > highest_similarity:

                        highest_similarity = similarity

                        most_similar_file2 = file2

            if highest_similarity >= threshold:

                results.append((file1, most_similar_file2, highest_similarity))

    return results

import os
import librosa
import numpy as np
import csv
import re
def normalize_to_range(data, min_val=-1, max_val=1):
    data_min = np.min(data)
    data_max = np.max(data)
    return min_val + (data - data_min) * (max_val - min_val) / (data_max - data_min)

def extract_channel2_data(file_path,sr):
    rate, data = read_wav(file_path,sr)
    data = trim_to_4_minutes(rate, data)
    #data = normalize_to_range(data, -1, 1)
    return data

def create_csv(folder2,results, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'timeFrame', 'upper_belt'])
        for file1, file2, similarity in results:
            ##change channel in filename
            modified_file2 = re.sub(r'channel\d+', 'channel3', file2)
            channel2_data = extract_channel2_data(folder2+"/"+modified_file2,25)
            time_frame = channel2_data
            
            for time, value in zip(time_frame, channel2_data):
                writer.writerow([file1, time, value])

def read_and_normalize(file_path):
    df = pd.read_csv(file_path)
    normalized_data = {}
    names = df['filename'].unique()
    for filename in names:
        try:
            file_data = df[df['filename'] == filename]['upper_belt'].values
            # Convert to numeric, forcing errors to NaN
            file_data = pd.to_numeric(file_data, errors='coerce')
            # Drop NaN values
            file_data = file_data[~np.isnan(file_data)]
            if len(file_data) > 0:
                normalized_data[filename] = file_data
            else:
                print(f"No numeric data for {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        
    return normalized_data

def compare_files(file1, file2):
    data1 = read_and_normalize(file1)
    data2 = read_and_normalize(file2)
    
    correlations = {}
    for filename in data1:
        if filename in data2:
            print(len(data1[filename]))
            print(len(data2[filename]))

            correlation, _ = pearsonr(data1[filename], data2[filename])
            correlations[filename] = correlation
        else:
            correlations[filename] = None  # No matching file in the second dataset
    return correlations
# Define the folders containing the wav files
home = "/home/glenn/Downloads"
folder1 = home+'/ComParE2020_Breathing/wav'

folder2 = home+"/UCL_data/full_wav"



# Compare the wav files and get the results


results = [
    ("devel_00.wav", "speaker33_channel1_trial10.wav", 1.00),
    ("devel_01.wav", "speaker37_channel1_trial10.wav", 1.00),
    ("devel_02.wav", "speaker38_channel1_trial10.wav", 1.00),
    ("devel_03.wav", "speaker39_channel1_trial10.wav", 1.00),
    ("devel_04.wav", "speaker40_channel1_trial10.wav", 1.00),
    ("devel_05.wav", "speaker43_channel1_trial10.wav", 1.00),
    ("devel_06.wav", "speaker44_channel1_trial10.wav", 1.00),
    ("devel_07.wav", "speaker45_channel1_trial10.wav", 1.00),
    ("devel_08.wav", "speaker46_channel1_trial10.wav", 1.00),
    ("devel_09.wav", "speaker47_channel1_trial10.wav", 1.00),
    ("devel_10.wav", "speaker48_channel1_trial10.wav", 1.00),
    ("devel_11.wav", "speaker49_channel1_trial10.wav", 1.00),
    ("devel_12.wav", "speaker50_channel1_trial10.wav", 1.00),
    ("devel_13.wav", "speaker51_channel1_trial10.wav", 1.00),
    ("devel_14.wav", "speaker53_channel1_trial10.wav", 1.00),
    ("devel_15.wav", "speaker54_channel1_trial10.wav", 1.00),
    ("test_00.wav", "speaker12_channel1_trial10.wav", 1.00),
    ("test_01.wav", "speaker15_channel1_trial10.wav", 1.00),
    ("test_02.wav", "speaker16_channel1_trial10.wav", 1.00),
    ("test_03.wav", "speaker17_channel1_trial10.wav", 1.00),
    ("test_04.wav", "speaker24_channel1_trial10.wav", 1.00),
    ("test_05.wav", "speaker25_channel1_trial10.wav", 1.00),
    ("test_06.wav", "speaker26_channel1_trial10.wav", 1.00),
    ("test_07.wav", "speaker27_channel1_trial10.wav", 1.00),
    ("test_08.wav", "speaker28_channel1_trial10.wav", 1.00),
    ("test_09.wav", "speaker29_channel1_trial10.wav", 1.00),
    ("test_10.wav", "speaker30_channel1_trial10.wav", 1.00),
    ("test_11.wav", "speaker31_channel1_trial10.wav", 1.00),
    ("test_12.wav", "speaker32_channel1_trial10.wav", 1.00),
    ("test_13.wav", "speaker34_channel1_trial10.wav", 1.00),
    ("test_14.wav", "speaker35_channel1_trial10.wav", 1.00),
    ("test_15.wav", "speaker36_channel1_trial10.wav", 1.00),
    ("train_00.wav", "speaker1_channel1_trial10.wav", 1.00),
    ("train_01.wav", "speaker2_channel1_trial10.wav", 1.00),
    ("train_02.wav", "speaker3_channel1_trial10.wav", 1.00),
    ("train_03.wav", "speaker4_channel1_trial10.wav", 1.00),
    ("train_04.wav", "speaker5_channel1_trial10.wav", 1.00),
    ("train_05.wav", "speaker6_channel1_trial10.wav", 1.00),
    ("train_06.wav", "speaker7_channel1_trial10.wav", 1.00),
    ("train_07.wav", "speaker8_channel1_trial10.wav", 1.00),
    ("train_08.wav", "speaker9_channel1_trial10.wav", 1.00),
    ("train_09.wav", "speaker10_channel1_trial10.wav", 1.00),
    ("train_10.wav", "speaker11_channel1_trial10.wav", 1.00),
    ("train_11.wav", "speaker13_channel1_trial10.wav", 1.00),
    ("train_12.wav", "speaker14_channel1_trial10.wav", 1.00),
    ("train_13.wav", "speaker18_channel1_trial10.wav", 1.00),
    ("train_14.wav", "speaker19_channel1_trial10.wav", 1.00),
    ("train_15.wav", "speaker21_channel1_trial10.wav", 1.00),
    ("train_16.wav", "speaker22_channel1_trial10.wav", 1.00)
]

results = compare_wav_files(folder1, folder2)


# Print the results

for file1, file2, similarity in results:

    print(f"Files: {file1} and {file2} have a similarity of {similarity:.2f}")




# Create the CSV file
output_file = 'labels.csv'
create_csv(folder2,results, output_file)


path = "D:/Downloads/UCL_data/wav"
og_path = "D:/Downloads/ComParE2020_Breathing/lab"
# Example usage
file1 = 'labels.csv'
file2 = og_path + '/labels.csv'
correlations = compare_files(file1, file2)

for filename, correlation in correlations.items():
    if correlation is not None:
        print(f"Pearson correlation for {filename}: {correlation:.2f}")
    else:
        print(f"No matching file for {filename} in the second dataset.")
        
seconds = 60000
#plot_average_trial_time(directory_path)
#split_wav_files_in_directory(path,chunk_length_ms=round(4*60000))

#split_wav_files_in_directory(path,seconds)
#data = create_output(path+"/splitted")
# Print unique filenames and times


# Load the CSV files into DataFrames
# gen_labels = pd.read_csv('output.csv')
# og_labels = pd.read_csv(og_path + '/labels.csv', sep=',')
# print(og_labels.head())
# matches = match_upperbelt_data('output.csv', og_path + '/labels.csv')
# print("MATCHES")

# #Print the results
# for match in matches:
#     print(f"Generated Filename: {match[0]}, Original Filename: {match[1]}, Match Percentage: {match[2]:.2f}%")

# # Print column names to verify
# print(f"MATCHES {len(matches)}")

# print("Generated labels columns:", gen_labels.columns)
# print("Original labels columns:", og_labels.columns)

# # Use the correct column names
# unique_filenames_gen = gen_labels['filename'].unique()
# unique_times_gen = gen_labels['timeFrame'].unique()
# print("Gen Unique filenames:", len(unique_filenames_gen))
# print("Gen Unique times:", len(unique_times_gen))

# # Adjust the column name based on the actual column names in og_labels
# # For example, if the column is named 'file_name' instead of 'filename'
# unique_filenames_og = og_labels['filename'].unique()  # Adjust this line
# unique_times_og = og_labels['timeFrame'].unique()  # Adjust if necessary
# print("OG Unique filenames:", len(unique_filenames_og))
# print("OG Unique times:", len(unique_times_og))
# # Filter filenames that contain the word "test"
# test_filenames = og_labels[og_labels['filename'].str.contains('test', case=False, na=False)]

# # Get the unique filenames
# unique_test_filenames = test_filenames['filename'].unique()

# # Print the number of unique filenames
# print("Number of unique filenames containing 'test':", len(unique_test_filenames))
# # Print the length of each filename
# for filename in test_filenames['filename'].unique():
#     print(f"Filename: {filename}, Length: {len(og_labels[og_labels['filename'].str.contains(filename, case=False, na=False)])}")