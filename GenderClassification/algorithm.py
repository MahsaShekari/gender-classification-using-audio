# Libraries
import os, os.path
from pathlib import Path
import librosa.display
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import average_precision_score
from sklearn.metrics import plot_precision_recall_curve

# we consider specific number of samples and call it n_sample
n_sample = 600

# First var Male, Second var Female
male_portion = 0.5
female_portion = 1 - male_portion

target_dataset_name = 'dev-clean'

data_repo_path = "./LibriSpeech/dev-clean"


gender_txt_file_path = "./LibriSpeech/SPEAKERS.TXT"


def gender_extractor(txt_file_add, dataset_name):
    """ Function to extract the ids of person according to their gender """
    male_list = []
    female_list = []
    with open(gender_txt_file_path, 'r') as source_file:
        source_file_lines = source_file.readlines()

    for line_idx in range(12, len(source_file_lines)):
        line = source_file_lines[line_idx]
        line_elements = line.split()
        # to find the name of the dataset
        if line_elements[4] == target_dataset_name:
            # to check the gender of the person
            if line_elements[2] == 'F':
                female_list.append(line_elements[0])
            else:
                male_list.append(line_elements[0])
        else:
            continue
    return male_list, female_list


# They are lists of male and female ids
male_ids_list, female_ids_list = gender_extractor(txt_file_add=gender_txt_file_path, dataset_name=target_dataset_name)


def transformer(ndarry):
    """Transform function take an ndarray, calculate mean and Standard deviation
     of that and then flatten these values"""
    # Considering a list for collecting the mean and Standard deviation each feature
    y_mean = []
    y_std = []

    # Iterating over ndarray features which is first element of the ndarray shape
    for i in range(ndarry.shape[0]):
        mean_rowi = ndarry[i].mean()
        y_mean.append(mean_rowi)
        std_rowi = ndarry[i].std()
        y_std.append(std_rowi)

    matrix_b = np.array([y_mean, y_std])

    y_flatten = matrix_b.flatten()
    return y_flatten


def voicesPathsCollector(source_dir, gender_ids):
    """ Function collects voice paths by having the path of source directory and list of the ids
    that we want to collect their path"""
    # The list containing the IDs of the speakers which are the folder names
    person_ids_list = gender_ids

    # An empty list which it will contains voices paths which are in the section folders
    voices_paths_list = []

    # Iterating over a person speaker ids
    for person_id_idx in range(len(person_ids_list)):
        # Speaker path containing the chapter's directories
        speaker_path = Path.cwd().joinpath(source_dir, person_ids_list[person_id_idx])
        # The list containing the IDs of the sections by the current speaker which are the folder names
        sections_ids_list = os.listdir(speaker_path)

        # Iterating over a sections ids by the current speaker
        for sec_id_idx in range(len(sections_ids_list)):
            # Section path containing files
            section_path = Path.cwd().joinpath(speaker_path,sections_ids_list[sec_id_idx])
            # Iterating over files which are existing in section's folders
            for file in os.listdir(section_path):
                # Choose files with .flac extension
                if file.endswith(".flac"):

                    voice_path = Path.cwd().joinpath(section_path, file)
                    voices_paths_list.append(voice_path)

    return voices_paths_list


male_voices_paths_list = voicesPathsCollector(source_dir=data_repo_path, gender_ids=male_ids_list)

female_voices_paths_list = voicesPathsCollector(source_dir=data_repo_path, gender_ids=female_ids_list)

# To shuffle the paths. Used to mimic the random selection
random.shuffle(male_voices_paths_list)
random.shuffle(female_voices_paths_list)


#  A list for keeping ndarray flatten
all_samples_list = []
#  A list keeping gender, we consider male as 1 and female as 0
target_list=[]


#  iterate on each audio in male_voices_paths_list
for i in range(int(n_sample*male_portion)):
    audio_file = male_voices_paths_list[i]
    # load audio files with librosa
    signal, sr = librosa.load(audio_file)
    # Extracting mfccs features by using signal and sample rate
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=10, sr=sr)
    # By calling transform function and passing the mfccs ,flatten ndarray will be obtained
    flatt_func = transformer(mfccs)
    all_samples_list.append(flatt_func)
    # As far as male are considered 1, target list will filled by one in each for iteration
    target_list.append(1)


#  iterate on each audio in female_voices_paths_list
for i in range(int(n_sample*female_portion)):
    audio_file = female_voices_paths_list[i]
    # load audio files with librosa
    signal, sr = librosa.load(audio_file)
    # Extracting mfccs features by using signal and sample rate
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=10, sr=sr)
    # By calling transform function and passing the mfccs ,flatten ndarray will be obtained
    flatt_func = transformer(mfccs)
    all_samples_list.append(flatt_func)
    # As far as female are considered 0, target list will filled by zero in each for iteration
    target_list.append(0)

# converting list to tuple
all_samples_tuple = tuple(all_samples_list)
all_samples = np.vstack(all_samples_tuple)

target = np.array(target_list)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(all_samples, target, test_size=0.3, random_state=109) # 70% training and 30% test

# #------------------ SVM(linear)---------------------

# Create an svm Classifier with Linear kernel
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_svm = clf.predict(X_test)

print("------------------ SVM Linear---------------------")


# Model Accuracy: how often is the classifier correct?
accuracy_svm = metrics.accuracy_score(y_test, y_pred_svm)
print("Accuracy_svm:", accuracy_svm)

# Model Precision: what percentage of positive tuples are labeled as such?
precision_svm = metrics.precision_score(y_test, y_pred_svm)
print("Precision_svm:", precision_svm)

# Model Recall: what percentage of positive tuples are labelled as such?
recall_svm = metrics.recall_score(y_test, y_pred_svm)
print("Recall_svm:", recall_svm)

# f1_score:
f1_score_svm = metrics.f1_score(y_test, y_pred_svm)
print("F1_score_svm:",f1_score_svm)


tn_svm, fp_svm, fn_svm, tp_svm = metrics.confusion_matrix(y_test, y_pred_svm).ravel()
print("TP by using svm classifier:", tp_svm)
print("TN by using svm classifier:", tn_svm)
print("FP by using svm classifier:", fp_svm)
print("FN by using svm classifier:", fn_svm)

#------------------ SVM(nonlinear)---------------------

#Create a svm(nonlinear) Classifier
clf_gaussian = svm.SVC(kernel='rbf', gamma='scale')

#Train the model using the training sets
clf_gaussian.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_svm_gaussian = clf_gaussian.predict(X_test)

print("------------------ SVM Nonlinear---------------------")


# Model Accuracy: how often is the classifier correct?
accuracy_svm_gaussian = metrics.accuracy_score(y_test, y_pred_svm_gaussian)
print("Accuracy_svm_Nonlinear:",accuracy_svm_gaussian)

# Model Precision: what percentage of positive tuples are labeled as such?
precision_svm_gaussian = metrics.precision_score(y_test, y_pred_svm_gaussian)
print("Precision_svm_Nonlinear:",precision_svm_gaussian)

# Model Recall: what percentage of positive tuples are labelled as such?
recall_svm_gaussian = metrics.recall_score(y_test, y_pred_svm_gaussian)
print("Recall_svm_Nonlinear:",recall_svm_gaussian)

# f1_score:
f1_score_svm_gaussian = metrics.f1_score(y_test, y_pred_svm_gaussian)
print("F1_score_svm_Nonlinear:",f1_score_svm_gaussian)

tn_svm_gaussian, fp_svm_gaussian, fn_svm_gaussian, tp_svm_gaussian = metrics.confusion_matrix(y_test, y_pred_svm_gaussian).ravel()
print("TP by using svm Nonlinear classifier:", tp_svm_gaussian)
print("TN by using svm Nonlinear classifier:", tn_svm_gaussian)
print("FP by using svm Nonlinear classifier:", fp_svm_gaussian)
print("FN by using svm Nonlinear classifier:", fn_svm_gaussian)

#------------------ Naive_Bayes---------------------

# Create a Gaussian Classifier
gnb = GaussianNB()
# Train the model using the training sets
gnb.fit(X_train, y_train)

# Predict the response for test dataset
y_pred_gnb = gnb.predict(X_test)

print("------------------ Naive_Bayes---------------------")

# Model Accuracy: how often is the classifier correct?
accuracy_gnb = metrics.accuracy_score(y_test, y_pred_gnb)
print("Accuracy_gnb:",accuracy_gnb)

# Model Precision: what percentage of positive tuples are labeled as such?
precision_gnb = metrics.precision_score(y_test, y_pred_gnb)
print("Precision_gnb:",precision_gnb)

# Model Recall: what percentage of positive tuples are labelled as such?
recall_gnb = metrics.recall_score(y_test, y_pred_gnb)
print("Recall_gnb:",recall_gnb)

# f1_score:
f1_score_gnb = metrics.f1_score(y_test, y_pred_gnb)
print("F1_score_gnb:",f1_score_gnb)

tn_gnb, fp_gnb, fn_gnb, tp_gnb = metrics.confusion_matrix(y_test, y_pred_gnb).ravel()
print("TP by using Naive_Bayes classifier:", tp_gnb)
print("TN by using Naive_Bayes classifier:", tn_gnb)
print("FP by using Naive_Bayes classifier:", fp_gnb)
print("FN by using Naive_Bayes classifier:", fn_gnb)




# --Compute the average precision score and Plot the Precision-Recall curve-----


# -------------2-class Precision-Recall curve SVM(linear kernel)------------------
average_precision_svm = average_precision_score(y_test, y_pred_svm)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision_svm))

disp = plot_precision_recall_curve(clf, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve SVM(linear kernel): '
                   'AP={0:0.2f}'.format(average_precision_svm))

# ------------------2-class Precision-Recall curve SVM(gaussian kernel)---------

average_precision_svm_gaussian = average_precision_score(y_test, y_pred_svm_gaussian)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision_svm_gaussian))

disp = plot_precision_recall_curve(clf_gaussian, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve SVM(gaussian kernel): '
                   'AP={0:0.2f}'.format(average_precision_svm_gaussian))

# ------------------2-class Precision-Recall curve Naive Bayes---------

average_precision_gnb = average_precision_score(y_test, y_pred_gnb)
print('Average precision-recall score: {0:0.2f}'.format(
      average_precision_gnb))

disp = plot_precision_recall_curve(gnb, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve Naive Bayes: '
                   'AP={0:0.2f}'.format(average_precision_gnb))


