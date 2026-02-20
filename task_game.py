import numpy as np
import tkinter as tk
from tkinter import ttk, PhotoImage


## To Do: save/export choices and options
 #        add rxn times??

class TaskTrial:

    """
    Class that creates a choice set in a feature selection task
    """

    def __init__(self, p_index):
        self.p_index = p_index
        self.frame_options = [] # list of options presented to user in a given fram
        self.choices = []       # list of states chosen by user
        self.score = 0    # 
        self.last_correct = 0 # number of previous correct answers
        self.feature = (np.random.randint(0,3), np.random.randint(0,4)) # 0 is type, 1 is which one
        print(self.feature)

        self.root = tk.Tk()
        self.frame = ttk.Frame(self.root, padding=50)
        self.score_label = ttk.Label(self.frame, text = "SCORE: " + str(self.score))
        self.b0 = ttk.Button(self.frame)
        self.b1 = ttk.Button(self.frame)
        self.b2 = ttk.Button(self.frame)
        self.b3 = ttk.Button(self.frame)
        

        self.frame.grid(row=0, column=0)
        self.score_label.grid(column=0, row=0)
        self.b0.grid(column = 0, row = 1)
        self.b1.grid(column = 1, row = 1)
        self.b2.grid(column = 0, row = 2)
        self.b3.grid(column = 1, row = 2)

    def _choose_stim(self, i, correct, choice):
        print("CHOSEN!" + str(i))
        if i == correct:
            self.last_correct += 1
            if np.random.random(1)[0] <= self.p_index:
                self.score += 50
                self.score_label['text'] = "SCORE: " + str(self.score)
        else:
            self.last_correct = 0
        self.choices.append(choice)
        if self.last_correct < 3:
            self.root.after(50, self.create_frames)
        else:
            self.root.destroy()

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
        filepath = r"C:\Users\newtc\OneDrive\Desktop\WIN 26\NEUSCI 403\task_demo\stims\\"

        # establish click interactions
        col0, shape0, pat0 = stimuli_options[:, 0]
        self.b0.image = PhotoImage(file = filepath + str(col0) + str(shape0) + str(pat0) + '.png').subsample(3, 3)
        self.b0.config(image=self.b0.image, command = lambda: self._choose_stim(0, correct, (col0, shape0, pat0)))

        col1, shape1, pat1 = stimuli_options[:, 1]
        self.b1.image = PhotoImage(file = filepath + str(col1) + str(shape1) + str(pat1) + '.png').subsample(3, 3)
        self.b1.config(image=self.b1.image, command = lambda: self._choose_stim(1, correct, (col1, shape1, pat1)))

        col2, shape2, pat2 = stimuli_options[:, 2]
        self.b2.image = PhotoImage(file = filepath + str(col2) + str(shape2) + str(pat2) + '.png').subsample(3, 3)
        self.b2.config(image=self.b2.image, command = lambda: self._choose_stim(2, correct, (col2, shape2, pat2)))

        col3, shape3, pat3 = stimuli_options[:, 3]
        self.b3.image = PhotoImage(file = filepath + str(col3) + str(shape3) + str(pat3) + '.png').subsample(3, 3)
        self.b3.config(image=self.b3.image, command = lambda: self._choose_stim(3, correct, (col3, shape3, pat3)))
        
        # update frame_options and choices
        self.frame_options.append(stimuli_options)
        
        

test = TaskTrial(0.7)
test.create_frames()

test.root.mainloop()






