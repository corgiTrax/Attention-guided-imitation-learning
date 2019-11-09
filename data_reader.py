import os, re, threading, time
import numpy as np
from IPython import embed
from scipy import misc
import vip_constants as V
import copy


RESIZE_SHAPE = 84

#TODO
class Dataset(Object):
    train_imgs, train_lbl, train_fid, train_size, train_weight = None, None, None, None, None
    val_imgs, val_lbl, val_fid, val_size, val_weight = None, None, None, None, None
    def __init__(self, TRAIN_VAL_FILE):
        train_trials, val_trains = parse_train_val_file(TRAIN_VAL_FILE)
        #Read training data labels and images into memory
        train_imgs, train_lbl, train_fid, train_size, train_weight = read_data_from_list(train_trials)
        #Read validation data labels and images into memory
        val_imgs, val_lbl, val_fid, val_size, val_weight = read_data_from_list(val_trials)
    

    print ("Performing standardization (x-mean)...")
    self.standardize()
    print ("Done.")

    def standardize(self):
        #TODO: test
        self.mean = np.mean(self.train_imgs, axis=(0,1,2))
        self.train_imgs -= self.mean # done in-place --- "x-=mean" is faster than "x=x-mean"
        self.val_imgs -= self.mean


#TODO
class Dataset_PastKFrames(Dataset):
    pass



def parse_train_val_file(filename):
    '''train_val file is in the format of:
    repo_1_name train
    repo_2_name val...'''
    train_trials, val_trials = [], []
    with open(filename, 'r') as f:
        for line in f:
            repo = line.split()[0]
            if line.split()[1] == "train":
                train_trials.append(copy.deepcopy(repo))
            else:
                val_trials.append(copy.deepcopy(repo))

    return train_trials, val_trials


#TODO
def read_data_list(listOfFiles):
    if len(listOfFiles) == 0:
        print("Warning: No trials specified, please make sure you have training and validation trial names in the data file")
        sys.exit(1)

    for file in listOfFiles:
        img_repo = file + ".tar.bz2"
        label_file = file + ".txt"

    # Read data from label files

    # Read img data

        with open(label_file,'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("frame_id") or line == "":
                    continue #skip header
                entries = line.split(',')
                frame_ids.append(entries[0])
                weight = 1
                if entries[5] == "null":# no action frame
                    weight = 0
                    actions.append(-1)
                else:
                    actions.append(int(entries[5]))
                # handle gaze positions
                if entries[6] == "null":# no gaze frame
                    weight = 0
                    gaze.append((-1,-1))
                else:
                    # Note: this version uses only the last gaze position
                    gaze_x, gaze_y = float(entries[-2]), float(entries[-1])
                    if gaze_out_of_bound(gaze_x, gaze_y):
                        weight = 0
                        gaze.append((-1,-1))
                    else
                        # transform gaze into reshaped image coordinate
                        gazes.append((gaze_x, gaze_y))
                weights.append(weight)
                break

    print(frame_ids, actions, gazes, weights)


def gaze_out_of_bound(x,y):
    '''Check whether the gaze position is out of game screen; this is possible since monitor is larger than game screen'''
    return x < 0 or y < 0 or x > 160 or y > 210

if __name__ == '__main__':
    seaquest_data = Dataset("seaquest-small.spec")
