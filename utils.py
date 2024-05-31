# Authors : Debanjali Biswas
#           Theresa Schmidt (theresas@lst.uni-saarland.de)
#           Iris Ferrazzo (ferrazzo@coli.uni-saarland.de)

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
Utility functions
"""

# importing libraries
import os
import re
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from conllu import parse
from matplotlib import style
from constants import prediction_file
from ast import literal_eval
import pandas as pd
from conllu2crowd_topk import generate_pairs_experiment
from constants import test_folder, folder
from conllu2crowd_topk import Recipe
style.use("ggplot")


def generate_recipe_dict(recipe, action_list, device):
    """
    Generate List of Recipe Dictionary

    Parameters
    ----------
    recipe : List
        Conllu parsed file for recipe.
    action_list : List
        List of action ids in recipe.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    recipe_node_list : List of Dict
        List of Dictionary for every action in the recipe.
        Dictionary contains Action_id, Action, Parent_List, Child_List

    """

    # No Alignment case
    empty_dict = {
        "Action_id": 0,
        "Action": torch.empty(0),
        "Parent_List": [],
        "Child_List": [],
    }

    recipe_node_list = [empty_dict]

    for action_id in action_list:

        action_node, parent_list, child_list = generate_data_line(
            action_id, recipe, device
        )
        # Recipe Dictionary
        recipe_dict = {
            "Action_id": action_id,
            "Action": action_node,
            "Parent_List": parent_list,
            "Child_List": child_list,
        }

        # Append Dictionary to List
        recipe_node_list.append(recipe_dict)
        #print(recipe_node_list)

    return recipe_node_list


#####################################


def generate_elmo_embeddings(emb_model, tokenizer, recipe, device):
    """
    Generate Elmo Embeddings

    Parameters
    ----------
    emb_model : ElmoEmbedding object
        Elmo Embedding model from Flair.
    tokenizer : flair.data.Sentence object
        Flair Data Sentence.
    recipe : List
        Conllu parsed file for recipe.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    embedding_vectors : Dict
        Embedding dictionary for a particular Recipe;
            where keys are vector_lookup_list token_ids and values are their corresponding Elmo embeddings.
    vector_lookup_list : Dict
        Look up dictionary for a particular Recipe embeddings;
            where key is the Conllu file token 'id' and values are list of token_ids generated using Elmo.

    """
    
    
    recipe_text = ""
    recipe_text_list = []

    for line in recipe[0]:
        recipe_text += line["form"] + " "
        recipe_text_list.append(line["form"])

    recipe_text = recipe_text.rstrip()

    recipe_tokens = tokenizer(recipe_text)  # Flair Sentence Representation
    
    emb_model.embed(recipe_tokens) # Elmo Embeddings

    embedding_vector = {}

    for i, token in enumerate(recipe_tokens):

        embedding_vector[i] = token.embedding
    
    #print(embedding_vector)

    vector_lookup_list = {}
    
    for i, token in enumerate(recipe_tokens):

        vector_lookup_list[i + 1] = [i]
    
    #print(vector_lookup_list)

    return embedding_vector, vector_lookup_list
    

#####################################


def generate_bert_embeddings(emb_model, tokenizer, recipe, device):
    """
    Generate Bert Embeddings for a recipe

    Parameters
    ----------
    emb_model : BertModel object
            Bert Embedding Model from HuggingFace.
    tokenizer : BertTokenizer object
            Tokenizer.
    recipe : List
        Conllu parsed file for recipe.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    embedding_vectors : Dict
        Embedding dictionary for a particular Recipe;
            where keys are vector_lookup_list token_ids and values are their corresponding BERT embeddings.
    vector_lookup_list : Dict
        Look up dictionary for a particular Recipe embeddings;
            where key is the Conllu file token 'id' and values are list of token_ids generated using BERT tokenizer.

    """

    # TODO this is a more pythonic way for the code below
    # until recipe_tokens = ...
    # recipe_text_list = [line['form'] for line in recipe[0]]
    # recipe_text = " ".join(recipe_text_list)

    recipe_text = ""
    recipe_text_list = []

    for line in recipe[0]:
        recipe_text += line["form"] + " "
        recipe_text_list.append(line["form"])

    recipe_text = recipe_text.rstrip()

    recipe_tokens = tokenizer.encode(recipe_text)  # Tokenize Recipe
    recipe_tensor = torch.LongTensor([recipe_tokens]).to(device)

    emb_model.eval()

    with torch.no_grad():

        outputs = emb_model(recipe_tensor)

        hidden = outputs[2]

    token_embeddings = torch.stack(hidden, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    # Stores the token vectors
    token_vecs_sum = []

    # For each token in the recipe...
    for token in token_embeddings:

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-2:], dim=0)

        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)

    embedding_vector = {}

    for i, emb in enumerate(token_vecs_sum):

        embedding_vector[i] = emb

    vector_lookup_list = {}
    count = 1

    for i, token in enumerate(recipe_text_list):

        tokenize = tokenizer.tokenize(token)

        vector_lookup_list[i + 1] = list(range(count, count + len(tokenize)))

        count += len(tokenize)

    return embedding_vector, vector_lookup_list


#####################################


def fetch_parsed_recipe(recipe_filename):
    """
    Fetch conllu parsed file

    Parameters
    ----------
    recipe_filename : String
        Recipe Filename.

    Returns
    -------
    parsed_recipe : List
       Conllu parsed file to generate a list of sentences.

    """
    
    file = open(recipe_filename, "r", encoding="utf-8")  # Recipe file
    conllu_file = file.read()  # Reading recipe file

    parsed_recipe = parse(conllu_file)  # Parsed Recipe File
    
    return parsed_recipe


#####################################

def fetch_dish_train(dish, folder, alignment_file, recipe_folder_name, emb_model, tokenizer, device, embedding_name):
        """
        Reads in the data.
        Author: Theresa Schmidt

        Parameters
        ----------
        dish : String
            Dish name.
        folder : String
            Path to data folder.
        alignment_file : String
            Filename of the alignment files.
        recipe_folder_name : String
            What the subfolder with the recipes is called in the dish folder.
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        device : object
            torch device where model tensors are saved.
        embedding_name : String
            Either 'elmo' or 'bert'.

        Returns
        ----------
        dish_dict : dict
            Contains all information from this dish. Keys: recipe names. Values: dictionaries with keys "Embedding_Vectors", "Vector_Lookup_Lists", "Action_Dicts_List" and values according to fetch_recipe().

        dish_group_alignments : pd.DataFrame
            All alignments (token ID's) for the dish, grouped by pairs of recipe names.
        """

        # Recipe names for the dish
        data_folder = os.path.join(folder, dish)  # dish folder
        recipe_folder = os.path.join(data_folder, recipe_folder_name)  # recipe folder, e.g. data/dish-name/recipes
        recipe_list = os.listdir(recipe_folder)
        recipe_list = [recipe for recipe in recipe_list if not recipe.startswith(".")]

        # Read in recipes of the dish
        dish_dict = dict()
        for recipe in recipe_list:

            recipe_filename = os.path.join(recipe_folder, recipe)

            embedding_vectors, vector_lookup_lists, action_dicts_list = fetch_recipe_train(
                recipe_filename, emb_model, tokenizer, device, embedding_name,
            )

            recipe_name = recipe.split(".")[0]
            dish_dict[recipe_name] = {"Embedding_Vectors" : embedding_vectors, "Vector_Lookup_Lists" : vector_lookup_lists, "Action_Dicts_List" : action_dicts_list}

        # Gold Standard Alignments between all recipes for dish
        alignment_file_path = os.path.join(
            data_folder, alignment_file
        )  # alignment file, e.g. data/dish-name/alignments.tsv
        alignments = pd.read_csv(
            alignment_file_path, sep="\t", header=0, skiprows=0, encoding="utf-8"
        )
        # Group by Recipe pairs
        dish_group_alignments = alignments.groupby(["file1", "file2"])

        return dish_dict, dish_group_alignments


###########################################


def fetch_dish_test(dish, folder, recipe_folder_name, emb_model, tokenizer, device, embedding_name):
        """
        Reads in the data.
        Author: Theresa Schmidt

        Parameters
        ----------
        dish : String
            Dish name.
        folder : String
            Path to data folder.
        alignment_file : String
            Filename of the alignment files.
        recipe_folder_name : String
            What the subfolder with the recipes is called in the dish folder.
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        device : object
            torch device where model tensors are saved.
        embedding_name : String
            Either 'elmo' or 'bert'.

        Returns
        ----------
        dish_dict : dict
            Contains all information from this dish. Keys: recipe names. Values: dictionaries with keys "Embedding_Vectors", "Vector_Lookup_Lists", "Action_Dicts_List" and values according to fetch_recipe().

        dish_group_alignments : pd.DataFrame
            All alignments (token ID's) for the dish, grouped by pairs of recipe names.
        """

        # Recipe names for the dish
        sub_directories=[]

        data_folder = os.path.join(folder, dish)  # dish folder
        recipe_folder = os.path.join(data_folder, recipe_folder_name)  # recipe folder, e.g. data/dish-name/recipes
        recipe_list = os.listdir(recipe_folder)
        recipe_list = [recipe for recipe in recipe_list if not recipe.startswith(".")]

        sub_directories.append(recipe_folder)

        # Read in recipes of the dish
        dish_dict = dict()
        for recipe in recipe_list:
            #print(recipe)

            recipe_filename = os.path.join(recipe_folder, recipe)
            #print(recipe_filename)

            embedding_vectors, vector_lookup_lists, action_dicts_list, recipe_actionlist_dict = fetch_recipe_test(
                recipe_filename, emb_model, tokenizer, device, embedding_name,
            )

            recipe_name = recipe.split(".")[0]
            #print(recipe_name)
            dish_dict[recipe_name] = {"Embedding_Vectors" : embedding_vectors, "Vector_Lookup_Lists" : vector_lookup_lists, "Action_Dicts_List" : action_dicts_list, "Action_id_List":recipe_actionlist_dict[recipe_name]}
            #print(dish_dict)
            
        # (Not Gold) Alignments between all recipes for dish
        # 3) extract actions to pair
     
        recipes_ids={}
      
        for recipe in recipe_list:
            #print(recipe)
            value_ids=[]

            recipe_filename = os.path.join(recipe_folder, recipe)
            recipe_open=open(recipe_filename).readlines()

            for line in recipe_open:
                  #print(line)
                  columns = line.strip().split()
                  id = columns[0]
                  token = columns[1]
                  tag = columns[4]         
                  
                  if tag[0] == "B":
                     #print(id)
                     value_ids.append(int(id))

            recipes_ids[recipe]=value_ids

        #print(recipes_ids)
             

        # 1) pair recipes using function generate_pairs_experiment from conllu2crowd

        for recipe in recipe_list:
           #print(recipe)

           pairs = generate_pairs_experiment(recipe_folder,#os.path.join(recipe_folder,recipe) 
                                               r1_indexed=False,
                                               r1_shuffled=False,
                                               r2_indexed=True,
                                               r2_indices_from=1)
           #print(pairs)
    
           # 2) convert class objects to strings
           pairings=[]
           for pair in pairs:
              source=pair[0].name#+".conllu"
              target=pair[1].name#+".conllu"
              pair=(source,target)
              pairings.append(pair)
           #print(pairings)
           #exit()
           
           #for pair in pairings:
           #   recipes_of_dish.append(pair[0])
           #   print(recipes_ids[recipe_1])

           # 4) prepare replacement for dish_group_alignments
           #print("keys of recipes_ids are",recipes_ids.keys())
           #print("pairings are",pairings)
           alignments=[]
           #for recipe in recipes_ids.keys():
           #   print(recipe)
              #used_tokens=[]
           for pair in pairings:
              #print(pair[0])
              #if recipe==pair[0]:
              #      print("yes")
              #      exit()
              used_tokens=[]
              recipe_1= pair[0]+".conllu"
              recipe_2= pair[1]+".conllu"
              try:
                 for value in recipes_ids[recipe_1]:
                    if value not in used_tokens:
                       alignment=(pair[0],value,pair[1])
                       #alignment.append(recipe_1)
                       #alignment.append(value)
                       used_tokens.append(value)
                       #alignment.append(recipe_2)
                       alignments.append(list(alignment))
                 used_tokens=[]
              except KeyError:
                 print("No match was found for recipe", recipe_1)
        #print(alignments)
           
           
        # 5) execute replacement for dish_group_alignments
        # I need the format list of lists
          
        dish_alignments = pd.DataFrame(alignments, columns =["file1", "token1", "file2"])
        dish_group_alignments = dish_alignments.groupby(["file1", "file2"])

        # if you want to inspect the content of dish_group_alignments
        #for key,item in dish_group_alignments:
        #    print(dish_group_alignments.get_group(key), "\n\n")
        print("len 492 (pred input) ", len(dish_group_alignments), len(dish_alignments))

        return dish_dict, dish_group_alignments
            
                        

#####################################


def fetch_dish_test_insertion(dish, folder, recipe_folder_name, emb_model, tokenizer, device, embedding_name):
        """
        Reads in the data.
        Author: Theresa Schmidt

        Parameters
        ----------
        dish : String
            Dish name.
        folder : String
            Path to data folder.
        alignment_file : String
            Filename of the alignment files.
        recipe_folder_name : String
            What the subfolder with the recipes is called in the dish folder.
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        device : object
            torch device where model tensors are saved.
        embedding_name : String
            Either 'elmo' or 'bert'.

        Returns
        ----------
        dish_dict : dict
            Contains all information from this dish. Keys: recipe names. Values: dictionaries with keys "Embedding_Vectors", "Vector_Lookup_Lists", "Action_Dicts_List" and values according to fetch_recipe().

        dish_group_alignments : pd.DataFrame
            All alignments (token ID's) for the dish, grouped by pairs of recipe names.
        """

        # Recipe names for the dish
        sub_directories=[]
        data_folder = os.path.join(folder, dish)  # dish folder = ./data/baked_ziti
        #print(data_folder)
        recipe_folder = os.path.join(data_folder, recipe_folder_name)  # recipe folder, e.g. ./data/baked_ziti/recipes, data/dish-name/recipes
        #print(recipe_folder) 
        recipe_list = os.listdir(recipe_folder)
        recipe_list = [recipe for recipe in recipe_list if not recipe.startswith(".")] # ['baked_ziti_6.conllu', 'baked_ziti_7.conllu', 'baked_ziti_9.conllu', 'baked_ziti_8.conllu', 'baked_ziti_2.conllu', 'baked_ziti_10.conllu', 'baked_ziti_3.conllu', 'baked_ziti_0.conllu', 'baked_ziti_4.conllu', 'baked_ziti_1.conllu', 'baked_ziti_5.conllu']
        #print(recipe_list)
        sub_directories.append(recipe_folder)

        # Read in recipes of the dish
        dish_dict = dict()
        for recipe in recipe_list:
            #print(recipe)

            recipe_filename = os.path.join(recipe_folder, recipe)
            #print(recipe_filename)

            embedding_vectors, vector_lookup_lists, action_dicts_list, recipe_actionlist_dict = fetch_recipe_test(
                recipe_filename, emb_model, tokenizer, device, embedding_name,
            )

            recipe_name = recipe.split(".")[0]
            #print(recipe_name)
            dish_dict[recipe_name] = {"Embedding_Vectors" : embedding_vectors, "Vector_Lookup_Lists" : vector_lookup_lists, "Action_Dicts_List" : action_dicts_list, "Action_id_List":recipe_actionlist_dict[recipe_name]}
            #print(dish_dict)
            
        # (Not Gold) Alignments between all recipes for dish
        # 3) extract actions to pair
     
        recipes_ids={}
      
        for recipe in recipe_list:
            #print(recipe)
            value_ids=[]

            recipe_filename = os.path.join(recipe_folder, recipe)
            recipe_open=open(recipe_filename).readlines()

            for line in recipe_open:
                  #print(line)
                  columns = line.strip().split()
                  id = columns[0]
                  token = columns[1]
                  tag = columns[4]         
                  
                  if tag[0] == "B":
                     #print(id)
                     value_ids.append(int(id))

            recipes_ids[recipe]=value_ids

        #print(recipes_ids)
             

        # 1) pair recipes using function generate_pairs_experiment from conllu2crowd

        for recipe in recipe_list:
           #print(recipe)

           pairs = generate_pairs_experiment(recipe_folder,#os.path.join(recipe_folder,recipe) 
                                               r1_indexed=False,
                                               r1_shuffled=False,
                                               r2_indexed=True,
                                               r2_indices_from=1)
           #print(pairs)
    
           # 2) convert class objects to strings
           pairings=[]
           for pair in pairs:
              source=pair[0].name#+".conllu"
              target=pair[1].name#+".conllu"
              pair=(source,target)
              pairings.append(pair)
           #print("ORIGINAL", "           ", pairings)

           print(pairings)
           # switch the order of the each element in pairings
           # this will alow us to have reversed alignment scores
           # we will use them to define insertions costs (=deletions costs) and to refine substitution costs (=average of the two align. directions)
           pairings_insertion=[]
           for pair in pairings:
              recipe1=pair[1]
              recipe2=pair[0]
              pair_ins=(recipe1,recipe2)
              pairings_insertion.append(pair_ins)
           #print("INSERTION", "           ", pairings_insertion)   
           

           # 4) prepare replacement for dish_group_alignments
           #print("keys of recipes_ids are",recipes_ids.keys())
           #print("pairings are",pairings)
           alignments=[]
           #for recipe in recipes_ids.keys():
           #   print(recipe)
              #used_tokens=[]
           for pair in pairings_insertion:
              #print(pair[0])
              #if recipe==pair[0]:
              #      print("yes")
              #      exit()
              used_tokens=[]
              recipe_1= pair[0]+".conllu"
              recipe_2= pair[1]+".conllu"
              try:
                 for value in recipes_ids[recipe_1]:
                    if value not in used_tokens:
                       alignment=(pair[0],value,pair[1])
                       #alignment.append(recipe_1)
                       #alignment.append(value)
                       used_tokens.append(value)
                       #alignment.append(recipe_2)
                       alignments.append(list(alignment))
                 used_tokens=[]
              except KeyError:
                 print("No match was found for recipe", recipe_1)
        #print(alignments)
           
           
        # 5) execute replacement for dish_group_alignments
        # I need the format list of lists
          
        dish_alignments = pd.DataFrame(alignments, columns =["file1", "token1", "file2"])
        dish_group_alignments = dish_alignments.groupby(["file1", "file2"])

        # if you want to inspect the content of dish_group_alignments
        #for key,item in dish_group_alignments:
        #    print(dish_group_alignments.get_group(key), "\n\n")

        return dish_dict, dish_group_alignments

#######################################


def fetch_recipe_train(recipe_filename, emb_model, tokenizer, device, embedding_name):
    """
    Fetch List of recipe dictionary and Embedding vector dictionary

    Parameters
    ----------
    recipe_filename : String
        Recipe filename (relative path).
    emb_model : Embedding object
        Model.
    tokenizer : Tokenizer object
        Tokenizer.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    embedding_vectors : Dict
        Embedding dictionary for a particular Recipe;
            where keys are vector_lookup_list token_ids and values are their corresponding word embeddings (BERT/ELMO).
    vector_lookup_list : Dict
        Look up dictionary for a particular Recipe embeddings;
            where key is the Conllu file token 'id' and values are list of token_ids generated using BERT/ELMO tokenizer.
    action_dicts_list : List of Dict
        List of Dictionary for every action in the recipe.
        Dictionary contains Action_id, Action, Parent_List, Child_List.

    """

    parsed_recipe = fetch_parsed_recipe(recipe_filename) # Parsed Recipe File
    
    if(embedding_name == 'bert'):

        embedding_vector, vector_lookup_list = generate_bert_embeddings(
            emb_model, tokenizer, parsed_recipe, device
        )  # Embeddings for Recipe
    
    elif (embedding_name == 'elmo'):
        
        embedding_vector, vector_lookup_list = generate_elmo_embeddings(
            emb_model, tokenizer, parsed_recipe, device
        )  # Embeddings for Recipe
    
    # addition by Iris: create a dict with recipe names as keys and corresponding action list as value --> to use in fetch_dish

    recipe_actionlist_dict=dict()
    # chage file name to match the one used in fetch_dish
    recipe_filename=recipe_filename.split("/")[-1]
    recipe_filename=recipe_filename.split(".")[0]

    action_list = fetch_action_ids(parsed_recipe)  # List of actions in recipe
    recipe_actionlist_dict[recipe_filename]=action_list
    #print(recipe_actionlist_dict)
    action_dicts_list = generate_recipe_dict(
        parsed_recipe, action_list, device
    )  # List of Recipe dictionary

    return embedding_vector, vector_lookup_list, action_dicts_list#, recipe_actionlist_dict

#####################################

def fetch_recipe_test(recipe_filename, emb_model, tokenizer, device, embedding_name):
    """
    Fetch List of recipe dictionary and Embedding vector dictionary

    Parameters
    ----------
    recipe_filename : String
        Recipe filename (relative path).
    emb_model : Embedding object
        Model.
    tokenizer : Tokenizer object
        Tokenizer.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    embedding_vectors : Dict
        Embedding dictionary for a particular Recipe;
            where keys are vector_lookup_list token_ids and values are their corresponding word embeddings (BERT/ELMO).
    vector_lookup_list : Dict
        Look up dictionary for a particular Recipe embeddings;
            where key is the Conllu file token 'id' and values are list of token_ids generated using BERT/ELMO tokenizer.
    action_dicts_list : List of Dict
        List of Dictionary for every action in the recipe.
        Dictionary contains Action_id, Action, Parent_List, Child_List.

    """

    parsed_recipe = fetch_parsed_recipe(recipe_filename) # Parsed Recipe File
    
    if(embedding_name == 'bert'):

        embedding_vector, vector_lookup_list = generate_bert_embeddings(
            emb_model, tokenizer, parsed_recipe, device
        )  # Embeddings for Recipe
    
    elif (embedding_name == 'elmo'):
        
        embedding_vector, vector_lookup_list = generate_elmo_embeddings(
            emb_model, tokenizer, parsed_recipe, device
        )  # Embeddings for Recipe
    
    # addition by Iris: create a dict with recipe names as keys and corresponding action list as value --> to use in fetch_dish

    recipe_actionlist_dict=dict()
    # chage file name to match the one used in fetch_dish
    recipe_filename=recipe_filename.split("/")[-1]
    recipe_filename=recipe_filename.split(".")[0]

    action_list = fetch_action_ids(parsed_recipe)  # List of actions in recipe
    recipe_actionlist_dict[recipe_filename]=action_list
    #print(recipe_actionlist_dict)
    action_dicts_list = generate_recipe_dict(
        parsed_recipe, action_list, device
    )  # List of Recipe dictionary

    return embedding_vector, vector_lookup_list, action_dicts_list, recipe_actionlist_dict

#####################################


def fetch_split_action(action_token_id, parsed_recipe):
    """
    Fetch tagging actions for a particular split action node

    Parameters
    ----------
    action_token_id : Int
        Action Token Id.
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.

    Returns
    -------
    tagging_tokens : string
        Tagging action sequence for a particular action node.

    """

    tagging_tokens = []
    token_id = action_token_id

    if token_id >= len(parsed_recipe[0]):
        return tagging_tokens

    line = parsed_recipe[0][token_id]
    
    # Checking for intermediate action node
    while line["xpostag"].startswith("I"):

        tagging_tokens.append(line["id"])
        token_id += 1
        line = parsed_recipe[0][token_id]

    return tagging_tokens


#####################################


def fetch_action_node(action_token_id, parsed_recipe, device):
    """
    Fetch Action String for a particular action node

    Parameters
    ----------
    action_token_id : Int
        Action Token Id.
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    action : Tensor
        Action sequence for a particular action node.

    """

    action = [parsed_recipe[0][action_token_id - 1]["id"]]

    tagged_action = fetch_split_action(action_token_id, parsed_recipe)

    if tagged_action:
        action.extend(tagged_action)

    # action_tokens = tokenizer.encode(action) # Tokenize action node

    action_tokens = torch.LongTensor(action).to(device)

    return action_tokens


#####################################
def dependency_heads(deps):
    """
    Interprets the entry in the DEPS column.
    Returns a list of head indices.
    """
    head_list = list()

    """
    if deps:
        deps = re.split("[( )]", deps) # TODO: rather use literaleval
        unwanted = {"[", "]"}
        head_list = [int(l.split(",")[0]) for l in deps if l not in unwanted]
    """
    # TODO: test (as soon as recipes are non-tree graphs)
    if deps:
        deps = literal_eval(deps)
        head_list = [int(h) for h, d in deps]

    return head_list

def fetch_parent_node(action_token_id, parsed_recipe, device):
    """
    Fetch List of Children for a particular action

    Parameters
    ----------
    action_token_id : Int
        Action Token Id.
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    parent_list : List of Tensors
        List of parents for that particular action.

    """

    action_line = parsed_recipe[0][action_token_id - 1]

    parent_list = list()

    parent_id = action_line["head"]

    if parent_id != 0:

        parent = [parsed_recipe[0][parent_id - 1]["id"]]

        tagged_action = fetch_split_action(parent_id, parsed_recipe)

        if tagged_action:
            parent.extend(tagged_action)

        # parent_token = tokenizer.encode(parent) # Tokenize parent node

        parent_token = torch.LongTensor(parent).to(device)

        parent_list.append(parent_token)  # Append to parent_list

        other_parents = dependency_heads(action_line["deps"])  # Other parents not belonging to head

        if other_parents:

            for parent_id in other_parents:

                parent = [parsed_recipe[0][parent_id - 1]["id"]]

                tagged_action = fetch_split_action(parent_id, parsed_recipe)

                if tagged_action:
                    parent.extend(tagged_action)

                # parent_token = tokenizer.encode(parent)

                parent_token = torch.LongTensor(parent).to(device)

                parent_list.append(parent_token)

    return parent_list


#####################################


def fetch_child_node(action_token_id, parsed_recipe, device):
    """
    Fetch List of Children for a particular action

    Parameters
    ----------
    action_token_id : Int
        Action Token Id.
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    child_list : List of Tensors
        List of children for that particular action.
    """

    child_list = list()

    for line in parsed_recipe[0]:

        if line["xpostag"].startswith("B") and (line["head"] == action_token_id or action_token_id in dependency_heads(line["deps"])): 

            child_id = line["id"]
            child = [parsed_recipe[0][child_id - 1]["id"]]

            tagged_action = fetch_split_action(child_id, parsed_recipe)

            if tagged_action:
                child.extend(tagged_action)

            # child_token = tokenizer.encode(child) # Tokenize Child node

            child_token = torch.LongTensor(child).to(device)

            child_list.append(child_token)  # Append child to child_list

    return child_list


#####################################


def generate_data_line(action_token_id, parsed_recipe, device):
    """
    Generate action, parent list and child list for a particular action node

    Parameters
    ----------
    action_token_id : Int
        Action Token Id.
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    action : Tensor
        Action node.
    parent_list : List of Tensors
        List of parents for that particular action.
    child_list : List of Tensors
        List of children for that particular action.

    """

    action = fetch_action_node(
        action_token_id, parsed_recipe, device
    )  # Fetch Action node sequence

    parent_list = fetch_parent_node(
        action_token_id, parsed_recipe, device
    )  # Fetch List of parents for a particular action

    child_list = fetch_child_node(
        action_token_id, parsed_recipe, device
    )  # Fetch List of children for a particular action

    return action, parent_list, child_list


#####################################


def fetch_action_ids(parsed_recipe):
    """
    Fetch all action ids in a conllu parse file

    Parameters
    ----------
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.

    Returns
    -------
    indices : List
        List of all action ids in a conllu parse file.

    """

    indices = list()

    for line in parsed_recipe[0]:

        # Checking for Action node
        if line["xpostag"].startswith("B-A"):
            indices.append(line["id"])

    return indices


#####################################


def save_vocabulary(path, vocab):
    """
    Save Naive Model Vocabulary

    Parameters
    ----------
    path : string
        path for saving the vocab.
    vocab : Dict
        Dictionary containing the action pairs are keys and their corresponding frequencies as values.

    Returns
    -------
    None.

    """
    
    with open(path, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"Model saved to ==> {path}")
    

#####################################


def load_vocabulary(path):
    """
    Load Saved Naive model Vocabulary

    Parameters
    ----------
    path : string
        path for saving the vocab.
        
    Returns
    -------
    vocab : Dict
        Saved Dictionary containing the action pairs are keys and their corresponding frequencies as values.

    """
    
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    
    return vocab
    

#####################################
    

def save_checkpoint(save_path, model, optimizer, valid_loss, valid_accuracy):
    """
    Function to save model checkpoint.

    Parameters
    ----------
    save_path : string
        path for saving the model checkpoints.
    model : object
        model.
    optimizer : object
        optimizer.
    valid_loss : float
        Validation loss.
    valid_accuracy : float
        validation accuracy.

    Returns
    -------
    None.

    """

    if save_path == None:
        return

    state_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
    }

    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")


#####################################


def load_checkpoint(load_path, model, optimizer, device):
    """
    Function to load model checkpoint.

    Parameters
    ----------
    load_path : string
        path for load the model checkpoints.
    model : object
        model.
    optimizer : object
        optimizer.
    device : object
        torch device where model tensors are saved.


    Returns
    -------
    model : AlignmentModel object
        Model after training
    optimizer : Adam optimizer object
        Optimizer after training.
    state_dict['valid_accuracy'] : Float
        Validation accuracy from the loaded state value dictionary.

    """

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f"Saved Model loaded from <== {load_path}")

    model.load_state_dict(
        state_dict["model_state_dict"],
        strict=False,
    )
    optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    return model, optimizer, state_dict["valid_accuracy"]


#####################################


def save_metrics(
    save_path,
    train_loss_list,
    valid_loss_list,
    train_accuracy_list,
    valid_accuracy_list,
    epoch_list,
):
    """
    Function to save model metrics.

    Parameters
    ----------
    save_path : string
        path for save the model metrics.
    train_loss_list : list
        list of training loss.
    valid_loss_list : list
        list of validation loss.
    train_accuracy_list : list
        List of traininbg accuracies
    valid_accuracy_list : list
        list of validation accuracies
    epoch_list : list
        list of epoch steps.

    Returns
    -------
    None.

    """

    if save_path == None:
        return

    state_dict = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "train_accuracy_list": train_accuracy_list,
        "valid_accuracy_list": valid_accuracy_list,
        "epoch_list": epoch_list,
    }

    torch.save(state_dict, save_path)
    print(f"Metrics saved to ==> {save_path}")


#####################################


def load_metrics(load_path, device):
    """
    Function to load model metrics.

    Parameters
    ----------
    load_path : string
        path for load the model metrics.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    dict
        metric dictionary containing training loss/accuracy and validation loss/accuracy at each epoch

    """

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f"Saved Metrics loaded from <== {load_path}")

    return state_dict


#####################################


def save_predictions(destination_folder, results_df, dish):
    """
    Save the predictions from the model during testing

    Parameters
    ----------
    destination_folder : string
        Destination folder name.
    results_df : Dataframe
        Results dataframe.
    dish : string
        dish name.

    Returns
    -------
    None.

    """

    save_prediction_path = os.path.join(
        destination_folder, dish + "_" + prediction_file
    )

    # Saving the results
    results_df.to_csv(save_prediction_path, sep="\t", index=False, encoding="utf-8")

    print("Predictions for Dish {} saved to ==> {}".format(dish, save_prediction_path))


#####################################


#def create_acc_loss_graph(file_path, device, save_graph_path):
def create_acc_loss_graph(epoch_list, train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list, device, save_graph_path):
    """
    Generate Training/Validation Accuracy and Loss Graph

    Parameters
    ----------
    file_path : String
        path for load the model metrics.
    device : object
        torch device where model tensors are saved.
    save_graph_path : String
        Path for saving the graph.

    Returns
    -------
    None.

    """

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    ax1.plot(
        epoch_list, train_loss_list, label="train_loss"
    )
    ax1.plot(
        epoch_list, valid_loss_list, label="valid_loss"
    )
    ax1.legend(loc=2)

    ax2.plot(
        epoch_list,
        train_accuracy_list,
        label="train_accuracy",
    )
    ax2.plot(
        epoch_list,
        valid_accuracy_list,
        label="valid_accuracy",
    )
    ax2.legend(loc=2)

    plt.show()

    ax1.autoscale()
    ax2.autoscale()
    fig.savefig(save_graph_path, dpi=fig.dpi)

    
    
'''dish_list = os.listdir(folder)
dish_list = [dish for dish in dish_list if not dish.startswith(".")]
dish_list.sort()
print(dish_list)

dish = dish_list[0]

transitive_property(folder, dish)'''
