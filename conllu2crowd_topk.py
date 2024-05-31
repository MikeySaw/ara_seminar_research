# Author: Theresa Schmidt, 2021 <theresas@coli.uni-saarland.de>

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Creates lists for crowdsourcing task with LingoTurk from collections of CoNLL-U files.
Expecting a directory where there is one folder per recipe type containing several
related recipes (path relative to the current directory). Recipes should be tsv
files with columns token-id, token, _, _, tag where actions are labelled with tags starting with B-A
and I-A (IOB2 format) in the tag column.

A list contains several slides which contain several questions. Each slide is a line in the list file.
A list is a tsv file with the following columns (for n=2 questions per slide):
        recipe1     full text of recipe 1
        recipe2     full text of recipe 2
        question1   full text of question 1
        question2   full text of question 2
        options1    list with options for question 1 with escaped quotations in colour definitions
        options2    list with options for question 2 with escaped quotations in colour definitions
        indices1    list with token indices corresponding to each first token in options1
        indices2    list with token indices corresponding to each first token in options2
        q1_id       token index of the first token of the action in question1
        q2_id       token index of the first token of the action in question2
        documentid1 document ID of recipe1
        documentid2 document ID of recipe2
        dish_name   name of the dish of the recipes
        slideid    slide id (unique for each run of main())
"""
import logging
import os
import random
from collections import defaultdict
from copy import copy, deepcopy
import argparse
import ast
import csv

# Colour definitions
target_colours = [
    "green",
    "red",
    "slateblue",
    "sienna",
    "hotpink",
    "gray",
    "olive",
    "dodgerblue",
    "limegreen",
    "salmon",
    "rebeccapurple",
    "chocolate",
    "deeppink",
    "darkgray",
    "darkkhaki",
    "teal",
    "cornflowerblue",
    "orange",
    "lightgreen",
    "tomato",
    "mediumpurple",
    "sandybrown",
    "mediumvioletred",
    "lightgray",
    "olivedrab",
    "powderblue",
    "navy",
    "bisque",
    "springgreen",
    "cyan",
    "yellowgreen",
    "indianred",
    "plum",
    "darkgoldenrod",
    "magenta",
    "darkslategray",
    "greenyellow",
    "aquamarine",
    "lightsteelblue",
    "rosybrown",
    "palegreen",
    "coral",
    "darkseegreen",
    "goldenrod",
    "burlywood",
    "lightpink",
    "peru",
    "saddlebrown",
    "palevioletred",
    "deepskyblue",
]

source_colours = ["blue", "darkorange", "seagreen", "firebrick"]


class Recipe:
    """
    Super class of SourceRecipe and TargetRecipe
    """

    # Collect metrics about all recipes
    recipe_lengths = dict()
    action_lengths = defaultdict(int)
    source_recipe_lengths = dict()

    def __init__(self, filename, tokens, eventindices, indexed=False, indices_from=1):
        """
        Arguments:
            - filename : str
            - tokens : list(str) Tokenized recipe text.
            - eventindices : list(list(int)) All token indices describing the actions in the recipe text.
            - indexed : bool Whether to assign index numbers to the actions in the recipe.
            - indices_from : Only applicable if indexed==True; defines the index of the first action.
        """
        self.name = ".".join(os.path.basename(filename).split(".")[:-1]) #filename.split("\\")[-1][:-7]
        self.dish_name = "_".join(self.name.split("_")[:-1])
        self.tokens = tokens
        self.eventindices = eventindices
        if indexed:
            # Index every event token:
            self.eventsequences = self.index_events(eventindices, indices_from)
        else:
            self.eventsequences = [
                self.get_text_at_indeces(span) for span in self.eventindices
            ]
        self.eventmapping = (
            self.generate_eventmapping()
        )  # a dictionary for formatting, e.g. indices, colours
        # maps start index of a sequence to sequence text
        self.full_text = None

        # heuristic over recipe lengths
        Recipe.recipe_lengths[self.name] = len(self.eventindices)
        # heuristic over action sequence lengths
        for seq in eventindices:
            Recipe.action_lengths[len(seq)] += 1

    def index_events(self, eventindices, start_index):
        """
        Each event/action gets a unique index number. Counting started from start_index s.t.
        we can have unique indices over a sequence of Recipes.

        Arguments:
            - eventindices : list(list(int)) All token indices describing the actions in the recipe text.
            - start_index : Only applicable if indexed==True; defines the index of the first action.
        """
        return [
            f"[{i} {self.get_text_at_indeces(span)}]"
            for i, span in enumerate(eventindices, start=start_index)
        ]

    def get_sequence_from_index(self, index, eventmapping=None):
        # Returns text at token index "index". If "index" is the first token of a multi-token action,
        # the whole action text is returned.
        if eventmapping and (index in eventmapping):
            return eventmapping[index]
        elif index in self.eventmapping:
            if eventmapping:
                logging.warning("used self.eventmapping even though an argument eventmapping was provided")
            return self.eventmapping[index]
        else:
            return self.tokens[
                int(index) - 1
            ]  # CoNLL-U indices start counting at 1, lists start counting at 0

    def get_text_at_indeces(self, indeces):
        """
        Returns one string made up of all the tokens referenced by the token indices in "indices".
        """
        text = ""
        for index in indeces:
            text += self.tokens[int(index) - 1] + " "
        return text[:-1]

    def generate_eventmapping(self):
        """
        Takes self.eventindices and self.eventsequences.
        Returns a dictionary mapping a token index to a text representation.
           The first token index in a sequence is mapped to the sequence's text representations.
           The other indices (if applicable) are mapped to the empty string.
        """
        mapping = defaultdict(str)  # result variable

        # Sanity check
        if len(self.eventindices) != len(self.eventsequences):
            raise RuntimeError(
                "Something went wrong. Eventindices and eventsequences should have the same length."
                "No mapping possible."
            )

        for i, span in enumerate(self.eventindices):
            # Mapping first token to full sequence string
            mapping[int(span[0])] = self.eventsequences[i]
            # Mapping rest to ""
            for idx in span:
                mapping[int(idx)] += ""
        return mapping

    def generate_single_string(self, coloured, eventmapping=None):
        """
        Arguments:
            - coloured : dict() A dictionary that maps token indices into string representations.
            Overrides other style definitions.
        Returns:
            Recipe text with actions printed in bold (and with indices and/or coloured, if specified earlier).
        Side-effect:
            Saves the recipe text in self.full_text
        """
        s = ""
        for i, t in enumerate(self.tokens, start=1):
            if i in coloured:
                if coloured[i]:  # in other words: if coloured[i] != ""
                    s += f"{coloured[i]} "
            elif eventmapping and (i in eventmapping):
                if eventmapping[i]:
                    s += f"<b>{eventmapping[i]}</b> "
            elif i in self.eventmapping:
                if self.eventmapping[i]:
                    s += f"<b>{self.eventmapping[i]}</b> "
                    if eventmapping:
                        logging.warning("using self.eventmapping even though there is a eventmapping!=None")
            else:
                s += f"{self.get_sequence_from_index(i)} "
        self.full_text = s
        return s

    def get_full_text(self, coloured=dict()):
        """
        Returns a string with the fully formatted recipe text.
        """
        if self.full_text:
            return self.full_text
        else:
            self.full_text = self.generate_single_string(coloured)
            return self.full_text

    def get_number_of_events(self):
        if self.eventindices:
            return len(self.eventindices)
        else:
            # No events (happens from time to time because there are broken recipes in the corpus)
            raise ValueError("No events in this recipe ", self.name)


class SourceRecipe(Recipe):
    """
    Source recipes have different actions highlighted in each question;
    each action should appear in exactly one question.
    Also, they can have different style definitions than target recipes.
    """

    def __init__(
        self,
        filename,
        tokens,
        eventindices,
        indexed=False,
        indices_from=1,
        shuffle_questions=False,
    ):
        """
        Arguments:
            - filename : str
            - tokens : list(str) Tokenized recipe text.
            - eventindices : list(list(int)) All token indices describing the actions in the recipe text.
            - indexed : bool Whether to assign index numbers to the actions in the recipe.
            - indices_from : Only applicable if indexed==True; defines the index of the first action.

            - shuffle_questions : bool If False, the questions will appear in
            the same order as the corresponding actions appear in the recipe text.
        """
        super().__init__(
            filename, tokens, eventindices, indexed=indexed, indices_from=indices_from
        )
        self.shuffle = shuffle_questions
        # Copy list of colours in order to be able pop() from it.
        self.colours = copy(source_colours)
        self.uncoloured_eventmapping = deepcopy(
            self.eventmapping
        )  # because self.eventmapping changes with every call of get_random_events()
        self.unseen_eventindices: list = self.eventindices

    def get_random_events(self, n, coloured=False):
        # Uses up the source events step by step
        """
        Returns a sample of n actions in the source recipes (random or orderly selection).
        Remembers which actions have been used already and only returns novel actions.

        Returns:
            - sample : list(list(Any))List of actions represented themselves as lists of token indices; len(sample)=n
        """

        # Get sample
        if self.shuffle:
            if n == -1:
                # Get all events
                n = len(self.eventindices)
            sample = random.sample(self.unseen_eventindices, n)
        else:
            if n == -1:
                # Get all events
                sample = deepcopy(self.eventindices)
            else:
                sample = deepcopy(self.unseen_eventindices[:n])

                # Sanity check
                if len(sample) != n and len(sample) != 1:
                    raise KeyError("len(sample) != n", self.unseen_eventindices, n, sample)

        # Delete sampled events from self.unseen_eventindices
        self.unseen_eventindices = [
            idx for idx in self.unseen_eventindices if idx not in sample
        ]
        if coloured:
            # Reset self.eventmapping
            self.eventmapping = deepcopy(self.uncoloured_eventmapping)

            # Put n events into self.eventmapping with respective colour
            if len(self.colours) < len(sample):
                raise ValueError("Not enough colours for this sample size.")
            for idx, c in zip(sample, self.colours):
                representation = self.eventmapping[int(idx[0])]
                if representation:  # Not empty
                    # Assign colour
                    self.eventmapping[
                        int(idx[0])
                    ] = f"<span style='color:{c}'>{representation}</span>"
                else:
                    self.eventmapping[int(idx[0])] = (
                        f"<span style='color:{c}'>"
                        f"{self.get_sequence_from_index(int(idx[0]))}</span>"
                    )
        
        return sample

    def get_full_text(self, coloured=set()):
        """
        Returns a string with the fully formatted recipe text.
        """

        # Has to be generated a new with every call because which actions are highlighted changes over time.
        self.full_text = self.generate_single_string(coloured)
        #print(self.full_text)
        return self.full_text


class TargetRecipe(Recipe):
    """
    The target recipe does not have to keep track of which actions have already been used.
    Its random sampling is different to that of source recipes. The formatting can
    also differ between source and target recipes.
    """

    def __init__(self, filename, tokens, eventindices, indexed=False, coloured=False, indices_from=1):
        """
        Arguments:
            - filename : str
            - tokens : list(str) Tokenized recipe text.
            - eventindices : list(list(int)) All token indices describing the actions in the recipe text.
            - indexed : bool Whether to assign index numbers to the actions in the recipe.
            - coloured : bool Whether the actions should be displayed with colours.
            - indices_from : Only applicable if indexed==True; defines the index of the first action.
        """
        super().__init__(
            filename, tokens, eventindices, indexed=indexed, indices_from=indices_from
        )
        if coloured:
                self.colours_dict = self.generate_colouring(copy(target_colours))
        else:
            self.colours_dict = None
        self.sample = []


    def generate_colouring(self, colours):
        """
        Generates a colour mapping where each action phrase / event sequence is mapped to a colour. Important so that in each slide for the same recipe pair, phrases always have the same colour.        

        Arguments:
            - colours : List    List of colour names.

        Returns:
            - colouring : dict
        """
        colouring = dict(zip([x[0] for x in self.eventindices], colours))

        return colouring


    def get_random_events(self, n, stored_sample=True, shuffled=True):
        """
        Returns a sample of n actions. Per default, the same sample is returned with each call of the function.

        Arguments:
            - n : int Number of actions to be returned.
            - stored_sample : bool Whether to return the same sample as in the previous call of the method.
            - shuffled : bool Whether the options should be displayed in the same
                              order as they appear in in the recipe text.

        Returns:
            - sample : list(list(Any)) List of actions represented themselves as lists of token indices; len(sample)=n
        """
        raise NotImplemented("changed constructor to define colour mapping self.colour_dict instead of self.colours")
        if stored_sample and self.sample:
            # sanity check (n=-1 means 'full sample')
            if len(self.sample) != n and n>0:
                logging.warning(
                    f"Requested sample size {n} could not be met with stored sample."
                )
            return deepcopy(self.sample)

        else:
            if shuffled:
                if n == -1:
                    # Get all events
                    n = len(self.eventindices)
                sample = random.sample(self.eventindices, n)
            else:
                if n == -1:
                    # Get all events
                    sample = deepcopy(self.eventindices)
                else:
                    sample = deepcopy(self.eventindices[:n])
                    if len(sample) != n and len(sample) != 1:
                        raise KeyError("len(sample) != n", self.eventindices)

            # Uses the target event colours
            if coloured:
                # Put n events into eventmapping with respective colour
                for idx in sample:
                    representation = self.eventmapping[int(idx[0])]
                    if representation.startswith("<s"):
                        # Colour already assigned
                        pass
                    elif representation:
                        # Not empty
                        self.eventmapping[
                            int(idx[0])
                        ] = f"<span style='color:{self.colours.pop(0)}'>{representation}</span>"
                    else:
                        self.eventmapping[int(idx[0])] = (
                            f"<span style='color:{self.colours.pop(0)}'>"
                            f"{self.get_sequence_from_index(int(idx[0]))}</span>"
                        )
            self.sample = copy(sample)
            return sample

    def expand_top_k_list(self, predicted):
        """
        Takes a list of first token indices and returns a list of lists where each inner list contains all token indices of a sequence.
        Input list will not be sorted!

        Arguments:
            - predicted : List(int) !! Be careful: predicted could contain token indices of first token in each action OR the indeces in predicted could mean n-th action in that specific recipe !!

        Returns:
            - l : List(List(int))
        """

        # remove 0
        # (0 means null alignment, i.e. there is no corresponding target action to be found in self.eventindices)
        # using while loop because there could be no 0's (or maybe even several?)
        while 0 in predicted:
            predicted.remove(0)

        #print("########\npredicted",predicted, self.name)
        #print(self.eventindices)
        l = [None]*len(predicted)

        for action in self.eventindices:
            #print("l", l)
            #print("\naction", action)
            if action[0] in predicted:
                l[predicted.index(action[0])] = action

        if None in l:
            raise RuntimeError("Could not find all elements of predicted in self.eventindices. Make sure that the predicted indices point are token indices (not indices pointing to the n-th action in a recipe).")
        else:
            return l

    def get_events_top_k(self, predicted, eventmapping, coloured=False, shuffled=False, sorting=False):
        """
        Returns a sample of actions containing exactly those actions that are specified by the prediction.

        Arguments:
            - predicted : list(int) List of indices each pointing to the first token of an action.
            - shuffled : bool Whether the order in which the options are displayed should be randomised. 
                              (Default: options appear in the order of the ranking as in predicted.)
            - sorting : bool  Whether the actions should be displayed in the same order as they appear in in the recipe text.
                              (Default: options appear in the order of the ranking as in predicted.)
            - eventmapping      Copy of self.eventmapping (don't want to change self.eventmapping)
            - coloured : bool

        Returns:
            - sample : list(list(Any)) List of actions represented themselves as lists of token indices; len(sample)=len(predicted)
            - eventmapping :            Style dictionary with colours added
        """
        # adjust ordering of the answer options
        if sorting:
            if shuffled: # sanity check
                raise RuntimeError("Options can either be sorted or shuffled - not both!")
            predicted = sorted(predicted)
            raise NotImplemented("double-check command line argument")
        if shuffled:
            random.shuffle(predicted) # shuffles predicted in place

        # retrieve all token indices of the answer options
        sample = self.expand_top_k_list(predicted) # List(List(int))

        # Uses the target event colours
        if self.colours_dict:
            # Put n events into eventmapping with respective colour
            for idx in sample:
                representation = eventmapping[int(idx[0])]
                if representation.startswith("<s"):
                    # Colour already assigned
                    pass
                elif representation:
                    # Not empty
                    eventmapping[
                        int(idx[0])
                    ] = f"<span style='color:{self.colours_dict[idx[0]]}'>{representation}</span>"
                else:
                    eventmapping[int(idx[0])] = (
                        f"<span style='color:{self.colours_dict[idx[0]]}'>"
                        f"{self.get_sequence_from_index(int(idx[0]))}</span>"
                    )
        self.sample = copy(sample)
        return sample, eventmapping

    def get_full_text(self, eventmapping):
        """
        Returns a string with the fully formatted recipe text.
        """

        # Has to be generated anew with every call because which actions are highlighted changes over time.
        self.full_text = self.generate_single_string(dict(), eventmapping=eventmapping)
        return self.full_text


def readfile(filename):
    """
    Read in recipe file in CoNLL-U format.
    Relevant columns: TOKEN-ID, FORM, _, _, TAG, _, _, _, _, _
    Action tags should start with B-A or I-A

    Returns:
        - filename : str Path to file.
        - tokens : list(str)
        - events : list(list(int)) List of lists of token indices for all actions in the recipe.
    """
    with open(filename, "r", encoding="utf-8") as f:
        # result variables
        tokens = []
        events = []
        # buffer variable
        event_acc = []

        # Check if the file is empty (there are broken files in the corpus).
        if os.stat(filename).st_size == 0:
            # discard the slide by raising an Error
            raise IOError(f"Empty file {filename}.")

        for line in f:
            # sometimes, there are two empty lines at the end of a file
            if line == "\n":
                break
            line = line.split("\t")
            tokens.append(line[1])

            # determine action sequences and store as lists of indexes in the variable events
            if line[4].startswith("B-A"):
                if event_acc != []:
                    events.append(event_acc)
                event_acc = [int(line[0])]
            elif line[4].startswith("I-A"):
                event_acc.append(int(line[0]))
            elif line[4] == "O":
                if event_acc != []:
                    events.append(event_acc)
                    event_acc = []
            else:
                # sanity check
                #raise RuntimeError(
                 #   f"Unexpected tag {line[4]} in for token {line[0]} in {filename}."
                #)
                # TODO: think about which version was better: only allowing tags B-A and I-A (previous) or allowing all labels starting with 'A' (current)
                if event_acc != []:
                    events.append(event_acc)
                    event_acc = []

        # if the last token was an action token , it's still in the buffer
        if event_acc != []:
            events.append(event_acc)

        return filename, tokens, events


class Question:
    """
    Class to keep track of the attributes of a question.
    """

    def __init__(self):
        self.text = ""
        self.options = ""
        self.options_indices = (
            ""  # token indices (within target recipe) for the answer options
        )
        self.question_index = ""  # token index of question action within its recipe
        self.unique_question_id = ""  # unique id of this question

    def set_options(self, tgt_events, tgt_sequences):
        """
        Transforms question attributes into string format.
        """
        
        # Define answer options
        self.options_indices = tgt_events
        self.options = tgt_sequences
        # Add escaping to colour definitions in the options as required by LingoTurk
        self.options = ["<b>" + s.replace("'", "\\'") + "</b>" for s in self.options]
        self.options.append("<b>None of these</b>")
        self.options = (
            "['" + "', '".join(self.options) + "']"
        )  # do like this to enforce escaping and single quotation marks in the appropriate places
        self.options_indices.append([0])  # for the none option
        self.options_indices = str(
            [int(l[0]) for l in self.options_indices]
        )  # remember the first token index for each option action


class Slide:
    """
    A slide can contain several questions from the same recipe pair. Several slides make a list.
    """

    slide_id = 0  # to generate original IDs

    def __init__(self, recipe1: SourceRecipe, recipe2: TargetRecipe):
        self.recipe1 = recipe1
        self.recipe2 = recipe2
        self.questions = []
        self.line = None
        self.slide_id = Slide.slide_id  # assign unique slide ID
        Slide.slide_id += 1

    def make_questions(
        self,
        n_questions,
        m_options,
        r1_coloured=True,
        r2_coloured=True,
        r2_shuffled=False,
    ):
        """
        n_questions: number of questions
        m_options: number of options (current state: all questions have the same options)
        """

        # Get n random events from recipe 1 as source actions (i.e. actions that we ask about in the question).
        # Sort by order of appearance in the recipe text
        src_events = sorted(
            self.recipe1.get_random_events(n_questions, coloured=r1_coloured),
            key=(lambda x: int(x[0])),
        )

        # Make questions for the n actions in src_events
        for event in src_events:
            q = Question()
            q.question_index = int(event[0])  # remember first token of source action
            self.questions.append(q)
            q.text = (
                f"Which action below from the second recipe does <b>"
                f"{self.recipe1.get_sequence_from_index(q.question_index)}"
                f"</b> from the first recipe correspond to best?"
            )

        # Get m random events from recipe 2 as targets
        try:
            tgt_events = self.recipe2.get_random_events(
                m_options, coloured=r2_coloured, shuffled=r2_shuffled
            )
        except ValueError:
            raise ValueError(
                f"Target recipe {self.recipe2.name} doesn't have enough events."
            )

        # get text corresponding to the indeces in tgt_events
        tgt_sequences = [
            self.recipe2.get_sequence_from_index(int(x[0])) for x in tgt_events
        ]

        # transform question attributes into string format
        for q in self.questions:
            q.set_options(tgt_events, tgt_sequences)

        # There may be less than n events left in the source recipe, therefore add as many empty questions as necessary.
        # Has to come after the above for-loop.
        for i in range(n_questions - len(src_events)):
            self.questions.append(Question())

        self.generate_text()  # So the colouring of recipe 1 is fixed

    def make_questions_top_k(
        self,
        top_k_dict,
        n_questions,
        r1_coloured=True,
        r2_coloured=True,
        r2_shuffled=False,
        r2_sorted=False
    ):
        """
        top_k_dict: maps source index to predicted target indices
        n_questions: number of questions
        """

        # Get n possibly random events from recipe 1 as source actions (i.e. actions that we ask about in the question) (randomness depends on recipe1.shuffle)
        # Sort by order of appearance in the recipe text
        src_events = sorted(
            self.recipe1.get_random_events(n_questions, coloured=r1_coloured),
            key=(lambda x: int(x[0])),
        )

        # Make questions for the actions in src_events
        tgt_eventmapping = deepcopy(self.recipe2.eventmapping)
        for event in src_events:
            
            q = Question()
            
            # Get target events from recipe 2
            try:
                tgt_events, tgt_eventmapping = self.recipe2.get_events_top_k(
                    top_k_dict[int(event[0])], tgt_eventmapping, coloured=r2_coloured, shuffled=r2_shuffled, sorting=r2_sorted
                )

                # get text corresponding to the indeces in tgt_events
                tgt_sequences = [
                    self.recipe2.get_sequence_from_index(int(x[0]), eventmapping=tgt_eventmapping) for x in tgt_events
                ]

                # transform question attributes into string format
                q.question_index = int(event[0])  # remember first token of source action
                q.text = (
                    f"Which action below from the second recipe does <b>"
                    f"{self.recipe1.get_sequence_from_index(q.question_index)}"
                    f"</b> from the first recipe correspond to best?"
                )
                q.set_options(tgt_events, tgt_sequences)
            except KeyError:
                logging.warning(f"Missing prediction for action {event} in recipe {self.recipe1.name}. Double-check input data to prediction with alignment model!")

            self.questions.append(q)

        # There may be less than n events left in the source recipe, therefore add as many empty questions as necessary.
        # Has to come after the above for-loop.
        for i in range(n_questions - len(src_events)):
            self.questions.append(Question())

        self.generate_text(tgt_eventmapping)  # So the colouring will be fixed

    def __str__(self):
        return self.get_text()

    def generate_text(self, tgt_eventmapping):
        """
        Compiles all the information of a slide into one string with tab-separated values.
        Saves the resulting string as self.line.

        Each line has the following fields: (for n=2)
        recipe1     full text of recipe 1
        recipe2     full text of recipe 2
        question1   full text of question 1
        question2   full text of question 2
        options1    list with options for question 1 with escaped quotations in colour definitions
        options2    list with options for question 2 with escaped quotations in colour definitions
        indices1    list with token indices corresponding to each first token in options1
        indices2    list with token indices corresponding to each first token in options2
        q1_id       token index of the first token of the action in question1
        q2_id       token index of the first token of the action in question2
        documentid1 document ID of recipe1
        documentid2 document ID of recipe2
        dish_name   name of the dish of the recipes
        slideid    slide id (unique for each run of main())
        """

        line = ""
        line += self.recipe1.get_full_text()
        line += "\t" + self.recipe2.get_full_text(tgt_eventmapping)
        for q in self.questions:
            line += "\t" + q.text
        for q in self.questions:
            line += "\t" + q.options
        for q in self.questions:
            line += "\t" + q.options_indices
        for q in self.questions:
            line += "\t" + str(q.question_index)
        line += "\t" + self.recipe1.name
        line += "\t" + self.recipe2.name
        line += "\t" + " ".join(
            [x[0].upper() + x[1:] for x in self.recipe1.name.split("_")[:-1]]
        )
        line += "\t" + str(self.slide_id)
        self.line = line

    def get_text(self):
        """
        Return all information of the Slide object as one tab-separated string.
        Each slide will be one line in the final list.
        """
        if self.line:
            return self.line
        else:
            self.generate_text()
            return self.line


def generate_pairs_experiment(
    subdir, r1_indexed=False, r1_shuffled=False, r2_indexed=False, r2_coloured=False, r2_indices_from=1
):
    """
    Go into subdir directory and read in all the recipes for the dish in subdir. Then pair source and target recipes
    s.t. the longest recipe (source) is paired with the next shorter recipe (target). The resulting list of
    recipe pairs has n-1 elements for n recipe files in the subdir directory.

    Returns: list of pairs of (SourceRecipe, TargetRecipe)
    """
    recipes_filenames = list(os.walk(subdir))[0][2]
    #print(recipes_filenames)
    src_recipes = []
    tgt_recipes = []

    # Read in all the recipes for the dish subdir
    for filename in recipes_filenames:
        file, tokens, events = readfile(os.path.join(subdir,filename))
        # Each recipe will be used as a source and target once
        src_recipes.append(
            SourceRecipe(
                file, tokens, events, indexed=r1_indexed, shuffle_questions=r1_shuffled
            )
        )
        tgt_recipes.append(
            TargetRecipe(
                file, tokens, events, indexed=r2_indexed, indices_from=r2_indices_from
            )
        )

    # Sort both lists by decreasing length (i.e. by number of events).
    # Secondary sorting: alphabetical
    src_recipes = sorted(
        src_recipes, key=(lambda x: (-x.get_number_of_events(), x.name))
    )
    
    tgt_recipes = sorted(
        tgt_recipes, key=(lambda x: (-x.get_number_of_events(), x.name))
    )
    #print(list(
    #    zip(src_recipes, tgt_recipes[1:])))
    return list(
        zip(src_recipes, tgt_recipes[1:])
    )  # pairing source recipes and target recipes; creating shifting window by offset 1

def read_predictions(dish_dir):
    """
    Reads in subdir/prediction.tsv expecting the columns 
        Recipe1, Action1_id, Recipe2, True_Label, Predicted_Label
    where Recipe1 and Recipe2 are recipe names, and Predicted_Label is a list of predicted integer token IDs.

    Returns dictionary with 
        - keys: pairs of recipe names (source, target)
        - values: dictionaries mapping Action1_id to list of predicted labels
    """
    pairs_dict = defaultdict(dict)
    """with open (os.path.join(dish_dir, 'prediction.tsv')) as f:
        for line in f:
            line = line.split("\t")
            pairs_dict[(line[0], line[2])][int(line[1])] = ast.literal_eval(line[4])"""

    with open(os.path.join(dish_dir, 'prediction.tsv')) as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            r1 = row["Recipe1"]
            r2 = row["Recipe2"]
            id1 = int(row["Action1_id"])
            pred = ast.literal_eval(row["Predicted_Label"])
            pairs_dict[(r1, r2)][id1] = pred

    return pairs_dict

def generate_lists_top_k(dish_dir, pairs_dict, questions_per_slide, r1_indexed=False, r1_shuffled=False, r2_indexed=False, r2_coloured=False, r2_indices_from=1):
    """
    
    """
    lists = [] # contains all slides for one dish; Slide objects associate one SourceRecipe object and one TargetRecipe object and contain n questions each
    for s,t in pairs_dict.keys():            
        s_file, s_tokens, s_events = readfile(os.path.join(dish_dir, 'recipes', s+".conllu"))
        source = SourceRecipe(s_file, s_tokens, s_events, indexed=r1_indexed, shuffle_questions=r1_shuffled)
        Recipe.source_recipe_lengths[
            source.name
        ] = (
            source.get_number_of_events()
        )  # To estimate the average length of source recipes

        t_file, t_tokens, t_events = readfile(os.path.join(dish_dir, 'recipes', t+".conllu"))
        target = TargetRecipe(t_file, t_tokens, t_events, indexed=r2_indexed, coloured=r2_coloured, indices_from=r2_indices_from)

        # Go over source recipe to create slides
        slides = [] 
        for i in range(
            int((source.get_number_of_events() + 1) / questions_per_slide)
        ):
            slide = Slide(source, target)
            # Make n questions with m options (n=questions_per_slide, m=n_options)
            slide.make_questions_top_k(
                pairs_dict[s,t],
                questions_per_slide,
                r1_coloured=args.r1_coloured,
                r2_coloured=args.r2_coloured,
                r2_shuffled=args.r2_shuffled,
                r2_sorted=args.r2_sorted
            )
            slides.append(slide)
        lists.append(slides)
   
    return lists

def halflist_pairing(half_lists):
    """
    Helper function for make_lists() 

    Returns:
            list of pairs of half_lists
    """
    lists = []

    # Sort by number of slides (i.e. sort by length) and secondarily sort alphabetically
    half_lists = sorted(
        half_lists, key=(lambda x: (len(x), x[0].recipe1.name))
    )  # x is a list of slides from the same recipe pair

    # Pair up the halflists s.t. longest is paired with shortest while making sure they are not for the same dish.
    while half_lists:
        slides1 = half_lists.pop()  # longest halflist
        slides2 = half_lists.pop(0)  # shortest halflist
        buffer = []
        while True:
            #(len(slides1), slides1[0].recipe1.name, len(slides2), slides2[0].recipe1.name, list(map(len,half_lists)))
            
            # Test whether slides1 and slides2 are for the same dish by comparing the first two letters of the name
            if slides1[0].recipe1.dish_name == slides2[0].recipe1.dish_name:
                buffer.append(slides1)  # remember the rejected recipe
                try:
                    slides1 = half_lists.pop()
                except IndexError:
                    if half_lists == []:
                        logging.warning(f"Pairing two recipe alignment pairs of the same dish: {(slides1[0].recipe1.name,slides1[0].recipe2.name)} and {(slides2[0].recipe1.name,slides2[0].recipe2.name)}. Fix manually if desired!")
                        break
                    else:
                        raise IndexError
            else:
                half_lists.extend(buffer)  # add rejected recipes to stack
                break
        lists.append((slides1, slides2))

    return lists


def make_lists(half_lists, args, out_dir="Lists"):
    """
    Takes the slides (arranged in half_lists) of all recipe pairs and
    sorts them into lists of two recipe pairs s.t. all resulting lists
    have approximately the same number of questions.

    Writes the resulting lists into separate files.

    Disclaimer: simplified and fitted to the current data - not tested against all eventualities
    """

    logging.info("\nLists: ")

    if args.two_dishes_per_list:
        lists = halflist_pairing(half_lists)
        logging.info("Number of lists: ", len(lists))
        logging.info(
            "Length of list, #actions in list1, #actions in list2, r1 of list1, r1 of list2:"
        )
        for i, pair in enumerate(lists):
            a, b = pair

            logging.info(
                f"{a[0].recipe1.get_number_of_events()+b[0].recipe1.get_number_of_events()} "
                f"{a[0].recipe1.get_number_of_events()} {b[0].recipe1.get_number_of_events()} "
                f"{a[0].recipe1.name} {b[0].recipe1.name}"
            )

            # Write lists to files
            try:
                with open(os.path.join(out_dir, "list_" + str(i) + ".tsv"), "w", encoding="utf-8") as f:
                    for slide in a:
                        f.write(slide.get_text() + "\n")
                    for slide in b:
                        f.write(slide.get_text() + "\n")
            except FileNotFoundError:
                raise FileNotFoundError(f"You may need to create the directory {out_dir} first.") 

    else:
        lists = half_lists
        logging.info("Number of lists: ", len(lists))
        logging.info(
            "#actions recipe 1, recipe 1, recipe 2:"
        )
        for i, slides in enumerate(lists):
            logging.info(
                f"{slides[0].recipe1.get_number_of_events()} {slides[0].recipe1.name} {slides[0].recipe2.name}"
            )

            # Write lists to files
            try:
                with open(os.path.join(out_dir, "list_" + str(i) + ".tsv"), "w", encoding="utf-8") as f:
                #with open(os.path.join(out_dir, "list.tsv"), "a", encoding="utf-8") as f:
                    for slide in slides:
                        f.write(slide.get_text() + "\n")
            except FileNotFoundError:
                raise FileNotFoundError(f"You may need to create the directory {out_dir} first.")


def main_round1(directory, out_dir, questions_per_slide=2, num_options=-1, args=None):
    """
    Assembles the recipe files in the subdirectories of directory into soure/target pairs
    and arranges them into questions on slides on lists.

    Arguments:
        - directory : str (Main directory (path relative to the current directory) with one
                           subdirectory per dishname; each subdirectory must contain at
                           least 2 recipes in separate files.)
        - out_dir : str (Output directory (path relative to the current directory) where the lists will be stored)
        - args : Namespace (style options)
            - r1_coloured : bool (default: True)
            - r1_indexed : bool (default: False)
            - r1_shuffled : bool (default: False)
            - r2_coloured : bool (default: True)
            - r2_indexed : bool (default: False)
            - r2_shuffled : bool (default: False)
    """
    raise NotImplemented("Made changes to command line arguments and some minor details s.t. this function needs to be double-checked and adapted before it can be run. Refer to the version on GitHub in the crowdsourcing repository for round 1 style lists.")
    # Define style options
    if args.r1_indexed and args.r2_indexed:
        r2_indices_from = 50
    else:
        r2_indices_from = 1

    dish_directories = [x[0] for x in os.walk(directory)][1:]
    half_lists = (
        []
    )  # a halflist consists of all slides of one recipe pair (a full list has two recipe pairs)
    for subdir in dish_directories:
        pairs = generate_pairs_experiment(
            subdir,
            r1_indexed=args.r1_indexed,
            r1_shuffled=args.r1_shuffled,
            r2_indexed=args.r2_indexed,
            r2_coloured=args.r2_coloured,
            r2_indices_from=r2_indices_from,
        )
        for source, target in pairs:
            Recipe.source_recipe_lengths[
                source.name
            ] = (
                source.get_number_of_events()
            )  # To estimate the average length of source recipes

            # Go over source recipe to create slides
            slides = []
            for i in range(
                int((source.get_number_of_events() + 1) / questions_per_slide)
            ):
                s = Slide(source, target)
                # Make n questions with m options (n=questions_per_slide, m=n_options)
                s.make_questions(
                    questions_per_slide,
                    num_options,
                    r1_coloured=args.r1_coloured,
                    r2_coloured=args.r2_coloured,
                    r2_shuffled=args.r2_shuffled,
                )
                slides.append(s)

            half_lists.append(slides)

    # Sort halflists into lists of two recipe pairs and write them into files
    make_lists(half_lists, args, out_dir)


def main_round2(directory, out_dir, questions_per_slide=2, num_options=-1, args=None):
    """
    Reads source/target pairs and top k predictions from directory/predctions.tsv and recipes from directory/recipes and arranges them into questions on slides on lists.

    Arguments:
        - directory : str (Main directory (path relative to the current directory) with one
                           subdirectory per dishname; each subdirectory must contain at
                           least 2 recipes in separate files.)
        - out_dir : str (Output directory (path relative to the current directory) where the lists will be stored)
        - args : Namespace (style options)
            - r1_coloured : bool (default: True)
            - r1_indexed : bool (default: False)
            - r1_shuffled : bool (default: False)
            - r2_coloured : bool (default: True)
            - r2_indexed : bool (default: False)
            - r2_shuffled : bool (default: False)
            - two_dishes_per_list : bool (default: False)
    """

    # Define style options
    if args.r1_indexed and args.r2_indexed:
        r2_indices_from = 50
    else:
        r2_indices_from = 1
    
    dish_directories = [x for x in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, x))]
    #print(dish_directories)
    half_lists = (
        []
    )  # a halflist consists of all slides of one recipe pair (a full list has two recipe pairs)
    for dish_dir in dish_directories:
        dish_dir = os.path.join(directory, dish_dir)
        print(dish_dir)
        #pairs = generate_pairs_experiment(subdir, r1_indexed=args.r1_indexed, r1_shuffled=args.r1_shuffled, r2_indexed=args.r2_indexed, r2_indices_from=r2_indices_from, )
        
        pairs_dict = read_predictions(dish_dir) # maps pairs of recipe names to dictionaries of predictions
        hl = generate_lists_top_k(dish_dir, pairs_dict, questions_per_slide, r1_indexed=args.r1_indexed, r1_shuffled=args.r1_shuffled, r2_indexed=args.r2_indexed, r2_coloured=args.r2_coloured, r2_indices_from=r2_indices_from)
        half_lists.extend(hl)

    # Write lists to files.
    # If required by args.two_dishes_per_list, sort halflists into lists of two recipe pairs and write them into files.
    make_lists(half_lists, args, out_dir)


if __name__ == "__main__":

    # parser for command line arguments
    arg_parser = argparse.ArgumentParser(
        description="""Creates lists for crowdsourcing task round 2 with LingoTurk from collections of CoNLL-U files and predictions for top k alignments between pairs of recipes."""
    )
    arg_parser.add_argument(
        "data",
        metavar="data",
        help="""Expecting path (relative to the 
        current directory) to a directory where there is one 
        sub-directory per dish. Each dish directory should contain a file 'prediction.tsv' and a sub-directory 'recipes' with several recipes. 
        Recipes should be tsv files with the columns TOKEN-ID, TOKEN, _, _, TAG 
        where actions are labelled with tags starting with B-A and I-A (IOB2 format) in the tag column.""",
    )
    arg_parser.add_argument(
        "-o",
        "--output-directory",
        dest="out",
        metavar="output_directory",
        default="Lists",
        help="Path relative to the current directory. Default: Lists",
    )
    arg_parser.add_argument(
        "-m",
        "--num-options",
        dest="m",
        default=-1,
        help="""Relevant for round 1. Maximum number of answer options per question (none-option is always added, 
        therefore the actual number of options is <=m+1). Default: no limit.""",
    )
    arg_parser.add_argument(
        "-n",
        "--num-questions",
        dest="n",
        default=2,
        help="""Number of questions per slide. Default: n=2""",
    ) # TODO: test different values for n
    arg_parser.add_argument(
        "--source-uncoloured",
        dest="r1_coloured",
        const=False,
        default=True,
        action="store_const",
        help="""Per default, the actions in the source recipe are printed bold and
         coloured (unique colour for each action).""",
    )
    arg_parser.add_argument(
        "--target-uncoloured",
        dest="r2_coloured",
        const=False,
        default=True,
        action="store_const",
        help="""Per default, the actions in the target recipe are printed bold and
         coloured (unique colour for each action).""",
    )
    arg_parser.add_argument(
        "--source-indexed",
        dest="r1_indexed",
        const=True,
        default=False,
        action="store_const",
        help="""Per default, the actions in the source recipe are not printed with 
        indices (unique indices).""",
    )
    arg_parser.add_argument(
        "--target-indexed",
        dest="r2_indexed",
        const=True,
        default=False,
        action="store_const",
        help="""Per default, the actions in the target recipe are not printed 
        with indices (unique indices).""",
    )
    arg_parser.add_argument(
        "--source-shuffled",
        dest="r1_shuffled",
        const=True,
        default=False,
        action="store_const",
        help="""Per default, the actions in the source recipe, and therefore 
        the questions, are presented in order of appearance.""",
    )
    arg_parser.add_argument(
        "--target-shuffled",
        dest="r2_shuffled",
        const=True,
        default=False,
        action="store_const",
        help="""The none option will still always be listed last. Per default, the actions in the target recipe are presented as 
        options in the same order as they appear in the recipe text (round 1) or ranked by predicted alignment likelihood, i.e. in the same order as in prediction.tsv (round 2). """,
    )
    arg_parser.add_argument(
        "--target-sorted",
        dest="r2_sorted",
        const=True,
        default=False,
        action="store_const",
        help="""Relevant for top k predictions - makes the options appear in the same order as they occur in the recipe text. Per default, the options are displayed in the order they come, i.e. ranked by predicted alignment likelihood. The none option will still always be listed last.""")
    arg_parser.add_argument(
        "--halflists",
        dest="two_dishes_per_list",
        const=True,
        default=False,
        action="store_const",
        help="""Per default, each list (i.e. output file) will contain one pair of recipes. In round 1 of the experiment, there were two such pairs of different dishes. Use this flag to get lists with two recipe pairs like in round 1.""")
    args = arg_parser.parse_args()

    #main_round1(
    #    args.data, args.out, questions_per_slide=args.n, num_options=args.m, args=args
    #)
    main_round2(
        args.data, args.out, questions_per_slide=args.n, num_options=args.m, args=args
    )

    print(
        f"\nThe length distribution for action sequences (total sum > total sum actions  because every recipe exists "
        f"two times - as a source and as a target): ",
        Recipe.action_lengths,
    )
    print(
        f"The longest action sequence has {max(Recipe.action_lengths.keys())} tokens."
    )
    print(
        f"The shortest action sequence has {min(Recipe.action_lengths.keys())} tokens."
    )
    print(
        f"The average action sequence has "
        f"{sum([x*y for x,y in Recipe.action_lengths.items()])/sum(Recipe.action_lengths.values())} tokens."
    )
    print(f"The longest recipe has {max(Recipe.recipe_lengths.values())} actions.")
    print(f"The shortest recipe has {min(Recipe.recipe_lengths.values())} actions.")
    print(
        f"The average recipe has {sum(Recipe.recipe_lengths.values())/len(Recipe.recipe_lengths)} actions."
    )
    print(
        f"The total number of action sequences (sum over all recipes)"
        f" is {sum(Recipe.recipe_lengths.values())}."
    )
    print(
        f"The average source recipe has "
        f"{sum(Recipe.source_recipe_lengths.values())/len(Recipe.source_recipe_lengths)} actions."
    )
    print(f"The longest source recipe has {max(Recipe.source_recipe_lengths.values())} actions.")
    print(f"The shortest source recipe has {min(Recipe.source_recipe_lengths.values())} actions.\n\n")
