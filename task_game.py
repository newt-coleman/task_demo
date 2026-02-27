import numpy as np
import tkinter as tk
from tkinter import ttk, PhotoImage
import pandas as pd
import os
import errno


## To Do: save/export choices and options
 #        add rxn times??

class TaskTrial:
    """
    Class that creates an interface for performing a trial of a feature selection task. Ends 
    the trial and exports data once a certain number of correct answers have been reached.

    Parameters
    ----------
    p_index : 
        float on the interval (0, 1.) that determines how probable receiving the reward is
        after selecting the correct answer.
    tk_root : 
        Tk root object that governs the event loop
    userid : 
        string of the file name the data will be exported to. Creates the file if it does
        not already exist.
    """

    THRESHOLD = 5

    def __init__(self, p_index, tk_root, userid):
        self.p_index = p_index
        self.userid = userid
        self.rewards = []       # 1 if choice was rewarded, zero otherwise

        choice_d = {"opt_0" : [], "opt_1" : [], "opt_2" : [], "opt_3" : [], "choice" : []}
        self.choice_data=pd.DataFrame(data=choice_d)

        self.score = 0    # 
        self.last_correct = 0 # number of previous correct answers
        self.feature = (np.random.randint(0,3), np.random.randint(0,4)) # 0 is type, 1 is which one
        print(self.feature)

        self.root = tk_root
        self.frame_start = ttk.Frame(self.root, padding=50)  # frame to start a trial
                                                             # breaks up individual trials

        self.frame_choose = ttk.Frame(self.root, padding=50) # main frame for choices
        self.score_label = ttk.Label(self.frame_choose, text = "SCORE: " + str(self.score))
        self.b0 = ttk.Button(self.frame_choose)
        self.b1 = ttk.Button(self.frame_choose)
        self.b2 = ttk.Button(self.frame_choose)
        self.b3 = ttk.Button(self.frame_choose)
        
        self.frame_start.grid(row=0, column=0)

        start_button = ttk.Button(self.frame_start, text="Begin new trial", command=self._start)
        start_button.grid(column = 0, row = 0)


    def _start(self):
        """
        Private method to switch from the start frame to the choice frame
        """
        self.frame_start.destroy()
        # self.frame_start.grid_forget()
        self.frame_choose.grid(row=0, column=0)

        self.score_label.grid(column=0, row=0)
        self.b0.grid(column = 0, row = 1)
        self.b1.grid(column = 1, row = 1)
        self.b2.grid(column = 0, row = 2)
        self.b3.grid(column = 1, row = 2)
        self.create_frames()

    def _choose_stim(self, i, correct, stimuli_options):
        """
        Callback for selecting a single stimuli. 

        Parameters
        ----------
        i : 
            position index of the chosen stimuli
        correct : 
            position index of the correct stimuli
        stimuli_options :
            array of the stimuli presented in the choice. Columns correspond
            to choices, and rows to features as in [colour, shape, pattern]
        
        """
        print("CHOSEN!" + str(i))
        if i == correct:
            self.last_correct += 1
            if np.random.random(1)[0] <= self.p_index: # check if should reward
                self.rewards.append(1)
                self.score += 50
                self.score_label['text'] = "SCORE: " + str(self.score)
            else:
                 self.rewards.append(0)
        else:  # only false ne
            self.rewards.append(0)
            self.last_correct = 0
        # else:  ## false positive
        #     if np.random.random(1)[0] >= self.p_index:
        #         self.rewards.append(1)
        #         self.score += 50
        #         self.score_label['text'] = "SCORE: " + str(self.score)
        #     else:
        #         self.rewards.append(0)
        #     self.last_correct = 0
            

        self.choice_data.loc[len(self.choice_data)] = [stimuli_options[:, 0], 
                                                       stimuli_options[:, 1], 
                                                       stimuli_options[:, 2], 
                                                       stimuli_options[:, 3], 
                                                       stimuli_options[:, i]]
        
        if self.last_correct < self.THRESHOLD:   # loop into next choice if not done
            self.root.after(50, self.create_frames)
        else: # export data and end this trial
            outpath = os.getcwd() + r"\\" + self.userid + r"\\"
            name = "choicedata_" + str(self.p_index) + "_" + str(self.feature[0]) + str(self.feature[1]) + "_"
            if os.path.exists(outpath) is False:
                 os.mkdir(outpath)
            unique = 0 
            while(os.path.isfile(outpath + name + str(unique) + '.csv')):
                 unique += 1
                
            self.choice_data.to_csv(outpath + name + str(unique) + '.csv')
            np.save(outpath + name + str(unique), self.rewards)
            self.frame_choose.destroy()

    def create_frames(self):
        """
        Creates a frame within a trial that has four options of mixed feature stimuli.
        """
        INDEX = np.array([0, 1, 2, 3])
        f_COLOR = ["cyan", 'green', 'magenta', 'yellow']
        f_SHAPE = ['circle', 'square', 'star', 'triangle']
        f_PATTERN = ['escher', 'polka', 'ripple', 'swirl']

        stimuli_options = np.array([np.random.permutation(INDEX),  # color
                                    np.random.permutation(INDEX),  # shape
                                    np.random.permutation(INDEX)]) # pattern
        correct = np.argwhere(stimuli_options[self.feature[0], :]==self.feature[1])
        print(correct)

        
        # find images
        filepath = os.getcwd() + r"\stims\\"
        if os.path.exists(filepath) is False:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

        # establish click interactions
        # these buttons callback to _choose_stim()
        col0, shape0, pat0 = stimuli_options[:, 0]
        self.b0.image = PhotoImage(file = filepath + str(col0) + str(shape0) + str(pat0) + '.png').subsample(3, 3)
        self.b0.config(image=self.b0.image, command = lambda: self._choose_stim(0, correct, stimuli_options))

        col1, shape1, pat1 = stimuli_options[:, 1]
        self.b1.image = PhotoImage(file = filepath + str(col1) + str(shape1) + str(pat1) + '.png').subsample(3, 3)
        self.b1.config(image=self.b1.image, command = lambda: self._choose_stim(1, correct, stimuli_options))

        col2, shape2, pat2 = stimuli_options[:, 2]
        self.b2.image = PhotoImage(file = filepath + str(col2) + str(shape2) + str(pat2) + '.png').subsample(3, 3)
        self.b2.config(image=self.b2.image, command = lambda: self._choose_stim(2, correct, stimuli_options))

        col3, shape3, pat3 = stimuli_options[:, 3]
        self.b3.image = PhotoImage(file = filepath + str(col3) + str(shape3) + str(pat3) + '.png').subsample(3, 3)
        self.b3.config(image=self.b3.image, command = lambda: self._choose_stim(3, correct, stimuli_options))
    
N_TRIALS = 10
P_LEVELS = [0.6, 0.8, 1.]


def _start_task(p_trials, root, userid):
     """
     Callback function to start sequence of trials
     """
     for child in root.winfo_children():
         child.destroy()

     for p in p_trials:
        test = TaskTrial(p, root, userid)

def run_trials(ntrials, p_levels, userid = 'test'):
    """
    Runs multiple trials of the feature detection task at varying deterministic levels
    """
    p_trials = []

    for p in p_levels:
            p_trials += [p]*ntrials
    p_trials = np.array(p_trials)
    np.random.shuffle(p_trials)

    root = tk.Tk()
    frame_info = ttk.Frame(root, padding=50)
    info_label = ttk.Label(frame_info, text = "This is a reasoning task that requires you to choose features to increase your score. " \
                                         "It is expected to take between x and y minutes. Press 'continue' to start. PLACEHOLDER ")
    start_button = ttk.Button(frame_info, text='Continue', command=lambda : _start_task(p_trials, root, userid))

    frame_info.grid(row=0, column=0)
    info_label.grid(row=0)
    start_button.grid(row=1)

   
    root.mainloop()

run_trials(N_TRIALS, P_LEVELS, )






