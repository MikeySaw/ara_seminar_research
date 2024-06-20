"""
Testing functions
"""

# importing libraries

import torch
import os
import flair
import argparse
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from model import AlignmentModel
from cosine_similarity_model import SimpleModel
from sequence_model import SequenceModel
from naive_model import NaiveModel
from transformers import BertTokenizer, BertModel
from flair.data import Sentence
from flair.embeddings import ELMoEmbeddings
from constants import OUTPUT_DIM, LR, MAX_EPOCHS, HIDDEN_DIM1, HIDDEN_DIM2, DROPOUT0, DROPOUT1, DROPOUT2, CUDA_DEVICE

from datetime import datetime
from constants import (
    folder,
    test_folder,
    alignment_file,
    recipe_folder_name,
    destination_folder1,
    destination_folder2,
    destination_folder3,
    destination_folder4,
)
from utils import (
    fetch_recipe_test,
    fetch_dish_test,
    save_metrics,
    save_checkpoint,
    load_checkpoint,
    save_predictions,
    create_acc_loss_graph,
    save_vocabulary,
    load_vocabulary
)

# -----------------------------------------------------------------------


# Testing Process Class
class Folds_Test:
    def run_model_test(
        self,
        dish_dict,
        dish_group_alignments,
        emb_model,
        tokenizer,
        model,
        device,
        embedding_name,
        criterion=None,
        optimizer=None,
        total_loss=0.0,
        step=0,
        correct_predictions=0,
        num_actions=0,
        mode="Training",
        model_name="Alignment Model",
    ):
        """
        Function to run the Model

        Parameters
        ----------
        dish_dict : dict
            Contains all information for one dish. Keys: recipe names. Values: dictionaries with keys "Embedding_Vectors", "Vector_Lookup_Lists", "Action_Dicts_List" and values according to fetch_recipe().
        dish_group_alignments : pd.DataFrame
            All alignments (token ID's) for one dish, grouped by pairs of recipe names.
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        model : AlignmentModel object
            Alignment model.
        device : object
            torch device where model tensors are saved.
        criterion : Cross Entropy Loss Function, optional
            Loss Function. The default is None.
        optimizer : Adam optimizer object, optional
            Optimizer. The default is None.
        total_loss : Float, optional
            Total Loss after Training/Validation. The default is 0.0.
        step : Int, optional
            Each Training/Validation step. The default is 0.
        correct_predictions : Int, optional
            Correction predictions for a Dish. Defaults is 0.
        num_actions : Int, optional
            Number of actions in a Dish. Defaults is 0.
        mode : String, optional
            Mode of Process - ("Training", "Validation", "Testing"). The default is "Training".
        model_name : String, optional
            Model name - ("Alignment Model", "Simple Model"). Default is "Alignment Model".


        """
        
        mode = "Testing"

        results_df = pd.DataFrame(
                columns=["Recipe1","Action1_id", "Recipe2", "Predicted_Label"]
            )

        #results_df = pd.DataFrame(columns=["Action1_id", "Predicted_Label"])
        # this was the original: (columns=["Action1_id", "True_Label", "Predicted_Label"])
                   
        for key in dish_group_alignments.groups.keys():
            
            recipe1 = dish_dict[key[0]] 
            recipe2 = dish_dict[key[1]] 

            recipe_pair_alignment = dish_group_alignments.get_group(key)
            #print(recipe_pair_alignment)

            #for node in action_dicts_list1[1:]:
            for node in recipe1["Action_Dicts_List"][1:]:
                #print(node)

                # True Action Id
                action_line = recipe_pair_alignment.loc[
                    recipe_pair_alignment["token1"] == node["Action_id"]
                ]
                #print(action_line)
                if not action_line.empty:
                 
                    # excluding part related to true label --> we evaluate later
                    #true_label = action_line["token2"].item()

                    # True Action Id index
                    #labels = [
                    #    i
                    #    for i, node in enumerate(recipe2["Action_Dicts_List"])
                    #    if node["Action_id"] == true_label
                    #]
                    #labels_tensor = torch.LongTensor([labels[0]]).to(device)

                    action1 = node["Action"]
                    parent_list1 = node["Parent_List"]
                    child_list1 = node["Child_List"]

                    # Generate predictions using our Alignment Model

                    if model_name == "Alignment Model":
                        prediction = model(
                            action1.to(device),
                            parent_list1,
                            child_list1,
                            recipe1["Embedding_Vectors"],
                            recipe1["Vector_Lookup_Lists"],
                            recipe2["Action_Dicts_List"],
                            recipe2["Embedding_Vectors"],
                            recipe2["Vector_Lookup_Lists"],
                        )

                    elif model_name == "Simple Model":
                        prediction = model(
                            action1.to(device),
                            recipe1["Embedding_Vectors"],
                            recipe1["Vector_Lookup_Lists"],
                            recipe2["Action_Dicts_List"],
                            recipe2["Embedding_Vectors"],
                            recipe2["Vector_Lookup_Lists"],
                        )

                    # print(prediction)

                    num_actions += 1

                    # Predicted Action Id
                    pred_label = recipe2["Action_Dicts_List"][torch.argmax(prediction).item()][
                        "Action_id"
                    ]

                    # here is evaluating --> we separate
                    #if true_label == pred_label:
                    #    correct_predictions += 1


                    results_dict = {
                                "Recipe1": key[0],
                                "Action1_id": node["Action_id"],
                                "Recipe2": key[1],
                                "Predicted_Label": pred_label, 
                            }

                    # Store the prediction
                    results_df = results_df.append(results_dict, ignore_index=True)
        print("num actions: ", num_actions)


        return correct_predictions, num_actions, results_df

        return None

    #####################################

    def test(self, dish_list, embedding_name, emb_model, tokenizer, model, destination_folder, device):
        """
        Test Function

        Parameters
        ----------
        dish_list : List
            List of dish names (typically, the list holds just one element).
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        model : AlignmentModel object
            Alignment model.
        destination_folder: String
            Destination folder.
        device : object
            torch device where model tensors are saved.

        Parameters
        ----------
        accuracy_list : List
            List of tuples (#correct predictions, #actions, dish accuracy) for each dish in dish_list.
        """

        mode = "Testing"

        accuracy_list = (
            []
        )  # List of tuples (#correct predictions, #actions, dish accuracy) for each dish in dish_list.

        for dish in dish_list:

            with torch.no_grad():

                correct_predictions, num_actions, results_df = self.run_model_test(
                    self.dish_dicts[dish], 
                    self.gold_alignments[dish], 
                    embedding_name = embedding_name,
                    emb_model=emb_model,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    mode=mode,
                )

            #print(correct_predictions)

            dish_accuracy = correct_predictions * 100 / num_actions
            
            save_predictions(destination_folder, results_df, dish)

            
            """
            # do evaluation with the same functions as in evaluate_predictions.py
            # TODO: add this paragraph in train.py

            # Notes: this paragraph should work; currently there is an error bc there are action in the recipe files that don't have an alignment in the alignment files which results in different lengths for gold and pred lists

            from sklearn.metrics import precision_recall_fscore_support, accuracy_score
            goldfile = os.path.join(folder, dish, alignment_file) # (abovementioned seld.gold_alignments seems to be missing column "token2" which would contain the gold data)

            # print(pd.read_csv(goldfile, sep="\t", encoding="utf-8").sort_values(by=["file1", "token1"]))
            # print(results_df)

            gold = pd.read_csv(goldfile, sep="\t", encoding="utf-8").sort_values(by=["file1", "token1"])#["token2"]
            pred = results_df.sort_values(by=["Recipe1", "Action1_id"])#["Predicted_Label"]

            print("len's 322 (gold,pred): ", len(gold), len(pred))
            gold.to_csv("gold.tsv", encoding="utf-8", index=False)
            pred.to_csv("pred.tsv", encoding="utf-8", index=False)

            sk_accuracy = accuracy_score(gold,pred)
            sk_prec, sk_recall, sk_f1, _ = precision_recall_fscore_support(gold , pred, average='weighted')

            accuracy_list.append([dish, correct_predictions, num_actions, dish_accuracy, sk_accuracy, sk_prec, sk_recall, sk_f1])
            """
            accuracy_list.append([dish, correct_predictions, num_actions, dish_accuracy, 0, 0, 0, 0])

        return accuracy_list

    #####################################


    def testing_process(
        self,
        dish_list,
        embedding_name,
        emb_model,
        tokenizer,
        model,
        optimizer,
        saved_file_path,
        saved_metric_path,
        destination_folder,
        device,
    ):
        """
        Testing Process function

        Parameters
        ----------
        dish_list : List
            List of all recipes in testing set (usually just 1).
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        model : AlignmentModel object
            Alignment model.
        optimizer : Adam optimizer object
            Optimizer.
        criterion : Cross Entropy Loss Function 
            Loss Function.  
        num_epochs : Int    
            Number of Epochs.
        saved_file_path : String
            Trained Model path.
        saved_metric_path : Sring
            Training Metrics file path.
        destination_folder: String
            Destination folder.
        device : object
            torch device where model tensors are saved.

        """

        model, optimizer, _ = load_checkpoint(saved_file_path, model, optimizer, device)

        # train_loss_list, valid_loss_list, epoch_list = load_metrics(saved_metric_path, device)


        accuracy_list = self.test(
            dish_list, embedding_name, emb_model, tokenizer, model, destination_folder, device
        )

        total_correct_predictions = 0
        total_actions = 0

        model.eval()

        for i, accuracy_line in enumerate(accuracy_list):
            # accuracy line: dish, correct_predictions, num_actions, dish_accuracy, sk_accuracy, sk_prec, sk_recall, sk_f1

            dish_accuracy = accuracy_line[3]

            total_correct_predictions += accuracy_line[1]
            total_actions += accuracy_line[2]

            #print("Accuracy on dish {} : {:.2f}".format(dish_list[i], dish_accuracy))

        model_accuracy = total_correct_predictions * 100 / total_actions

        #print("Accuracy on full test set: {:.2f}".format(model_accuracy))
        #print(f"Test set: {dish_list}")

        return accuracy_list, model_accuracy, total_correct_predictions, total_actions
    
    
#####################################

    def run_folds_test(
        self,
        embedding_name,
        emb_model,
        tokenizer,
        model,
        optimizer,
        criterion,
        num_epochs,
        device,
        with_feature=True,
    ):
        """
        Running 10 fold cross validation for alignment models

        Parameters
        ----------
        embedding_name : String
            Either 'elmo' or 'bert'.
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        model : AlignmentModel object
            Alignment model.
        optimizer : Adam optimizer object
        num_epochs : Int
            Number of Epochs.
        device : object
            torch device where model tensors are saved.
        with_feature : boolean; Optional
            Check whether to add features or not. Default value True.

        """

        fold = args.fold

        print("-------Loading Data-------")

        dish_list = os.listdir(folder)

        dish_list = [dish for dish in dish_list if not dish.startswith(".")]
        dish_list.sort() # sorting here has become important as we determine which dish is test_dish and valid_dish from the fold index

        train_dish_list = dish_list.copy()
        if fold in range(len(dish_list)):
            test_dish_id = fold  # Validation dish index
        else:
            test_dish_id = 0

        dish_list_test = [
                train_dish_list.pop(test_dish_id)
            ]
        print("Testing on dish", dish_list_test)

        dish_list_test = [dish for dish in dish_list_test if not dish.startswith(".")]
        dish_list_test.sort() # TODO: why?

        self.dish_dicts = dict()
        self.gold_alignments = dict()

        for dish in dish_list_test:
        
            dish_dict, dish_group_alignments = fetch_dish_test(dish, folder, recipe_folder_name, emb_model, tokenizer, device, embedding_name)
            #dish_dict: Keys: recipe names. Values: dictionaries with keys "Embedding_Vectors", "Vector_Lookup_Lists", "Action_Dicts_List"
            #dish_group_alignments: pd.DataFrame of alignment file
        
            self.dish_dicts[dish] = dish_dict
        
            self.gold_alignments[dish] = dish_group_alignments

        print("Data successfully loaded for test dishes ", dish_list_test)

        fold_result_df = pd.DataFrame(
            columns=[
                "Fold",
                "Test_Dish",
                "Accuracy1",
                "Accuracy2",
                "F1",
                "Precision",
                "Recall",
                "Correct_Predictions",
                "Num_Actions"
            ]
        )

        if with_feature:
            destination_folder = destination_folder1

        else:
            destination_folder = destination_folder2

        print("-------Cross Validation Folds-------")

        start = datetime.now()

        saved_file_path = os.path.join(
            destination_folder, "model" + str(fold) + ".pt"
        )  # Model saved path
        saved_metric_path = os.path.join(
            destination_folder, "metric" + str(fold) + ".pt"
        )  # Metric saved path
        saved_graph_path = os.path.join(destination_folder, "loss_acc_graph" + str(fold) + ".png")

        test_dish_list = dish_list_test

        if fold in range(len(dish_list)):
            test_dish_id = fold  # Validation dish index
        else:
            test_dish_id = 0

        print("Fold [{}/{}]".format(fold, len(dish_list)))

        print("-------Testing-------")

        (
            test_accuracy_list, # list of lists with the following values: dish, correct_predictions, num_actions, dish_accuracy, sk_accuracy, sk_prec, sk_recall, sk_f1
            test_accuracy,
            total_correct_predictions,
            total_actions,
        ) = self.testing_process(
            test_dish_list,
            embedding_name,
            emb_model,
            tokenizer,
            model,
            optimizer,
            saved_file_path,
            saved_metric_path,
            destination_folder,
            device,
        )

        end = datetime.now()

        elapsedTime = end - start
        elapsed_duration = divmod(elapsedTime.total_seconds(), 60)

        print(
            "Time elapsed: {} mins and {:.2f} secs".format(
                elapsed_duration[0], elapsed_duration[1]
            )
        )
        # print("test_dish_id +1, dish_list[test_dish_id] ", test_dish_id +1, dish_list[test_dish_id])
        # try:
        #    fold_result = {
        #        "Fold": fold + 1,
        #        "Train_Loss": train_loss,
        #        "Train_Accuracy": train_accuracy,
        #        "Valid_Loss": valid_loss,
        #        "Valid_Accuracy": valid_accuracy,
        #        "Test_Accuracy": test_accuracy,
        #        "Correct_Predictions": total_correct_predictions,
        #        "Num_Actions": total_actions,
        #        "Test_Dish": dish_list[test_dish_id+1],
        #        "Fold_Timelapse_Minutes": elapsed_duration[0]
        #    }  # ,
        # "Test_Dish1_accuracy" : test_accuracy_list[0][2],
        # "Test_Dish2_accuracy" : test_accuracy_list[1][2]}
        # except IndexError:
        for line in test_accuracy_list:
            fold_result = {
                "Fold": fold + 1,
                "Test_Dish" : line[0],
                "Accuracy1" : line[3],
                "Accuracy2" : line[4],
                "F1" : line[7],
                "Precision" : line[5],
                "Recall": line[6],
                "Correct_Predictions" : line[1],
                "Num_Actions" : line[2]}
            fold_result_df = fold_result_df.append(fold_result, ignore_index=True)

        print("--------------")

        save_result_path = os.path.join(destination_folder, "fold_results_test.tsv")

        # Saving the results
        if os.path.exists(save_result_path):
            fold_result_df.to_csv(save_result_path, sep="\t", index=False, encoding="utf-8", header=False, mode="a")
        else: 
            fold_result_df.to_csv(save_result_path, sep="\t", index=False, encoding="utf-8")

        print("Fold Results saved in ==>" + save_result_path)


# FUNCTIONS FOR OTHER MODELS: SIMPLE, SIMILARITY, ETC.
#-----------------------------------------------------------------------------------------------------
        

    def test_simple_model(self, embedding_name, emb_model, tokenizer, simple_model, device):
        """
        Testing Cosine Similarity Baseline

        Parameters
        ----------
        embedding_name : String
            Embedding name Bert/Elmo
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        simple_model : SimpleModel object
            Simple Baseline model.
        device : object
            torch device where model tensors are saved.

        """

        total_correct_predictions = 0
        total_actions = 0

        dish_list = os.listdir(folder)

        test_result_df = pd.DataFrame(columns=["Dish", "Correct_Predictions", "Num_Actions","Accuracy"])

        dish_list = [dish for dish in dish_list if not dish.startswith(".")]
        dish_list.sort()

        saved_file_path = os.path.join(
            destination_folder3, "model_result.tsv"
        )  # Model saved path

        for dish in dish_list:

            correct_predictions, num_actions, results_df = self.run_model_test(
                self.dish_dicts[dish], 
                self.gold_alignments[dish], 
                emb_model,
                tokenizer,
                simple_model,
                device,
                embedding_name,
                mode="Testing",
                model_name="Simple Model",
            )

            save_predictions(destination_folder3, results_df, dish)

            accuracy = correct_predictions * 100 / num_actions

            test_result = {
                "Dish": dish,
                "Correct_Predictions": correct_predictions,
                "Num_Actions": num_actions,
                "Accuracy": accuracy,
            }

            test_result_df = test_result_df.append(test_result, ignore_index=True)

            total_correct_predictions += correct_predictions
            total_actions += num_actions

        model_accuracy = total_correct_predictions * 100 / total_actions

        test_result = {
            "Dish": "Overall",
            "Correct_Predictions": total_correct_predictions,
            "Num_Actions": total_actions,
            "Accuracy": model_accuracy,
        }

        test_result_df = test_result_df.append(test_result, ignore_index=True)

        print("Model Accuracy: {:.2f}".format(model_accuracy))

        test_result_df.to_csv(saved_file_path, sep="\t", index=False, encoding="utf-8")

        print("Results saved in ==>" + saved_file_path)
        

#####################################
    
    
    def basic_testing(self,
                      model,
                      dish_list,
                      saved_file_path,
                      destination_folder,
                      test_result_df):
        
        total_correct_predictions = 0
        total_actions = 0
   
        
        
        vocab = load_vocabulary(saved_file_path) #load saved vocabulary
        #print(vocab)

        for dish in dish_list:
            data_folder = os.path.join(folder, dish)  # dish folder
            recipe_folder = os.path.join(data_folder, recipe_folder_name)  # recipe folder
    
            alignment_file_path = os.path.join(
                data_folder, alignment_file
            )  # alignment file
            
        
            # Gold Standard Alignments between all recipes for dish
               
            alignments = pd.read_csv(
                alignment_file_path, sep="\t", header=0, skiprows=0, encoding="utf-8"
            )
    
            # Group by Recipe pairs
            dish_group_alignments = alignments.groupby(["file1", "file2"])
            
            num_actions = 0
            correct_predictions = 0
            
            results_df = pd.DataFrame(
                columns=["Action", "Predicted_Label"]
            )
            
            for key in dish_group_alignments.groups.keys():

               recipe1_filename = os.path.join(recipe_folder, key[0] + ".conllu")
               recipe2_filename = os.path.join(recipe_folder, key[1] + ".conllu")
                
               recipe_pair_alignment = dish_group_alignments.get_group(key)
               
               _, parsed_recipe2, action_pairs = model.generate_action_pairs(recipe_pair_alignment, recipe1_filename, recipe2_filename)
               
               correct_predictions, num_actions, results_df = model.fetch_aligned_actions(action_pairs, 
                   vocab, 
                   parsed_recipe2,
                   correct_predictions,
                   num_actions,
                   results_df)
               
            total_correct_predictions += correct_predictions
            total_actions += num_actions
               
            save_predictions(destination_folder, results_df, dish)

            accuracy = correct_predictions * 100 / num_actions
            
            print("Dish Accuracy: {:.2f}".format(accuracy))

            test_result = {
                "Dish": dish,
                "Correct_Predictions": correct_predictions,
                "Num_Actions": num_actions,
                "Accuracy": accuracy,
                }
            
            test_result_df = test_result_df.append(test_result, ignore_index=True)
            
        model_accuracy = total_correct_predictions * 100 / total_actions
        
        print("Model Accuracy: {:.2f}".format(model_accuracy))

        return model_accuracy, total_correct_predictions, total_actions, test_result_df
        

#####################################


    def run_naive_folds_test( self,
        model
        ):
        """
        Running 10 fold cross validation for naive baseline

        Parameters
        ----------
        model : NaiveModel object
            Naive Baseline model

        """

        dish_list_test = os.listdir(folder_test)

        dish_list_test = [dish for dish in dish_list_test if not dish.startswith(".")]
        dish_list_test.sort()

        fold_result_df = pd.DataFrame(
            columns=[
                "Fold",
                "Test_Accuracy",
                "Correct_Predictions",
                "Num_Actions",
            ]
        )  # , "Test_Dish1_accuracy", "Test_Dish2_accuracy"])

        test_dish_id = len(dish_list_test)
        
        destination_folder = destination_folder4
        
        test_result_df = pd.DataFrame(columns=["Dish","Correct_Predictions","Num_Actions","Accuracy"])
        overall_predictions = 0
        overall_actions = 0 

        for fold in range(len(dish_list_test)):

            start = datetime.now()

            saved_file_path = os.path.join(
                destination_folder, "model" + str(fold + 1) + ".pt"
            )  # Model saved path

            #train_dish_list = dish_list.copy()
            test_dish_list = dish_list_test  #[
            #    train_dish_list.pop(test_dish_id)
            #]  # , train_dish_list.pop(test_dish_id - 1)]

            test_dish_id -= 1

            if test_dish_id == -1:

                test_dish_id = len(dish_list_test) - 1

            print("Fold [{}/{}]".format(fold + 1, len(dish_list_test)))

            print("-------Testing-------")

            (
                test_accuracy,
                total_correct_predictions,
                total_actions,
                test_result_df
            ) = self.basic_testing(
                model,
                test_dish_list,
                saved_file_path,
                destination_folder,
                test_result_df
            )
                
            overall_predictions += total_correct_predictions
            overall_actions += total_actions

            fold_result = {
                "Fold": fold + 1,
                "Test_Accuracy": test_accuracy,
                "Correct_Predictions": total_correct_predictions,
                "Num_Actions": total_actions,
            }  # ,
            # "Test_Dish1_accuracy" : test_accuracy_list[0][2],
            # "Test_Dish2_accuracy" : test_accuracy_list[1][2]}

            fold_result_df = fold_result_df.append(fold_result, ignore_index=True)

            end = datetime.now()

            elapsedTime = end - start
            elapsed_duration = divmod(elapsedTime.total_seconds(), 60)

            print(
                "Time elapsed: {} mins and {:.2f} secs".format(
                    elapsed_duration[0], elapsed_duration[1]
                )
            )
            print("--------------")
            
            
        overall_accuracy = overall_predictions * 100 / overall_actions
        
        print("Overall Model Accuracy: {:.2f}".format(overall_accuracy))
        
        fold_result = {
                "Fold": 'Overall',
                "Test_Accuracy": overall_accuracy,
                "Correct_Predictions": overall_predictions,
                "Num_Actions": overall_actions,
            }
        
        fold_result_df = fold_result_df.append(fold_result, ignore_index=True)

        save_result_path = os.path.join(destination_folder, "fold_results.tsv")
        
        results_file_path = os.path.join(
            destination_folder, "model_result.tsv"
        )  # Model saved path

        # Saving the results
        fold_result_df.to_csv(save_result_path, sep="\t", index=False, encoding="utf-8")
        
        test_result_df.to_csv(results_file_path, sep="\t", index=False, encoding="utf-8")

        print("Fold Results saved in ==>" + save_result_path)
 
# -------------------------------------------------------------------------------
if __name__ == "__main__":

    # from script main.py
    # no more function, merged ith train-related functions

    device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")
    flair.device = device
        
    parser = argparse.ArgumentParser(description = """Automatic Alignment model""")
    parser.add_argument('model_name', type=str, help="""Model Name; one of {'Simple', 'Naive', 'Alignment-no-feature', 'Alignment-with-feature'}""") # TODO: add options for fat graphs (with parents and grandparents)
    parser.add_argument('--embedding_name', type=str, default='bert', help='Embedding Name (Default is bert, alternative: elmo)')
    parser.add_argument('--cuda-device', type=str, default="0", help="""Select cuda; default: cuda:0""")
    parser.add_argument('--fold', type=int, default=1, help="""Fold Number; number in range 1 to 10""")
    args = parser.parse_args()

    model_name = args.model_name
        
    embedding_name = args.embedding_name

    if args.cuda_device:
        device = torch.device("cuda:"+args.cuda_device if torch.cuda.is_available() else "cpu")
        flair.device = device 

    fold = args.fold

    print("-------Loading Model-------")

    # Loading Model definition
        
    if embedding_name == 'bert' :

        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )  # Bert Tokenizer
        
        emb_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(
            device
        )  # Bert Model for Embeddings
            
        embedding_dim = emb_model.config.to_dict()[
            "hidden_size"
        ]  # BERT embedding dimension
        
        # print(bert)
        
    elif embedding_name == 'elmo' :
            
        tokenizer = Sentence #Flair sentence for ELMo embeddings
        
        emb_model = ELMoEmbeddings('small')
            
        embedding_dim = emb_model.embedding_length

       
    # final part of main.py

    TT = Folds_Test()  # calling the Training class

    if model_name == "Alignment-with-feature":

        model = AlignmentModel(embedding_dim, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, DROPOUT0, DROPOUT1, DROPOUT2, device).to(
            device
        )  # Out Alignment Model with features

        #print(model)
        """for name, param in model.named_parameters():
            if param.requires_grad:
                    print(name)"""

        optimizer = optim.Adam(model.parameters(), lr=LR)  # optimizer for training
        criterion = nn.CrossEntropyLoss()  # Loss function

        ################ Cross Validation Folds #################

        TT.run_folds_test(
            embedding_name, 
            emb_model, tokenizer, model, optimizer, criterion, MAX_EPOCHS, device
        )

    elif model_name == "Alignment-no-feature":

        model = AlignmentModel(
            embedding_dim, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, DROPOUT0, DROPOUT1, DROPOUT2, device, False
        ).to(
            device
        )  # Out Alignment Model w/o features

        print(model)

        optimizer = optim.Adam(model.parameters(), lr=LR)  # optimizer for training
        criterion = nn.CrossEntropyLoss()  # Loss function

        TT.run_folds_test(
            embedding_name,
            emb_model, 
            tokenizer,
            model,
            optimizer,
            criterion,
            MAX_EPOCHS,
            device,
            False,
        )

    elif model_name == "Cosine_similarity":

        cosine_similarity_model = SimpleModel(embedding_dim, device).to(device) # Simple Cosine Similarity Baseline

        print(cosine_similarity_model)

        print("-------Testing (Simple Baseline) -------")

        TT.test_simple_model(embedding_name, emb_model, tokenizer, cosine_similarity_model, device)
            
            
    elif model_name == 'Naive':
            
        naive_model = NaiveModel(device) # Naive Common Action Pair Heuristics Baseline
            
        print('Common Action Pair Heuristics Model')
            
        ################ Cross Validation Folds #################
            
        TT.run_naive_folds(
            naive_model
            )
            
    elif model_name == 'Sequence':
            
        sequence_model = SequenceModel()
            
        print('Sequential Alignments')
            
        sequence_model.test_sequence_model()

    else:

        print(
            "Incorrect Argument: Model_name should be ['Cosine_similarity', 'Naive', 'Alignment-no-feature', 'Alignment-with-feature']"
        )

