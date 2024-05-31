# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import csv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from constants import folder, alignment_file, test_folder, destination_folder1

sub_dirss= []
results_per_dish={}
#prediction_dir=destination_folder1
prediction_dir=sys.argv[1]

# extract root, sub directories, and all files of test folder
for root_name, sub_dirs, file_list in os.walk(test_folder):  # , topdown = False):
   #print("root_name", root_name)
   #for file in sub_dirs:
      #print(file)
   for sub_dir in sub_dirs:
      #print("sub_dir",sub_dir)
      if sub_dir != "recipes":
         # store subdirectories that are different from "recipes" --> first order ones
         sub_dirss.append(sub_dir)
   #print(sub_dirss)

   # go search for the alignment files
   for filename in file_list:
      #print("filenames", filename)
      for sub_dir in sub_dirss:
         if filename==alignment_file:
            #print(filename)
            align_file_open = open(os.path.join(folder, sub_dir, filename))#.read()
            align_file = align_file_open.readlines()[1:]
            gold_pred={}
            gold=[]

            # define elements for each line in alignment file 

            for line_align in align_file:
                #print("ALIGN",line_align)
                #print(line_align.strip().split())
                columns_align = line_align.strip().split()
                rec1 = columns_align[0]
                #print(rec1)
                act_id = columns_align[1]
                rec2 = columns_align[2]
                gold_label = columns_align[3]

                # dictionary with gold alignments
                # gold_pred.keys() = recipe 1, action id
                # gold_pred.values() = recipe 2, gold prediction
                gold_pred[(rec1,act_id)] = (rec2,gold_label) 
                #print (act_id, gold_label)
                #print("gold",gold_pred)

                # extract list of prediction files
                pred_files=[] 
                for pred_file in os.listdir(prediction_dir):
                   pred_files.append(pred_file)
                   #print(pred_file)
                   model_pred={}
                   pred=[]
                   tot_predictions = 0
                   correct_predictions = 0

                      
                   if os.path.isfile(os.path.join(prediction_dir, pred_file)):
                      #print(pred_file.split(".")[:-1][0])
                      #print(sub_dir+"_prediction")
                      #print(join(str(pred_file).split("_")[:-1]))
                      #if str(pred_file).split("_")[:-1][0]==sub_dir:
                      if pred_file.split(".")[:-1][0]==sub_dir+"_prediction":

                         model_pred={}
                         #print("yes",str(pred_file))

                         pred_file_open = open(os.path.join(prediction_dir, pred_file))#.read()
                         pred_file = pred_file_open.readlines()[1:]

                         # define elements for each line in prediction file
                         for line in pred_file:
                             #print("PRED",line)
                             #line=line.strip().split()
                             #print(line)
                             tot_predictions += 1
                             #print(tot_predictions)
                             columns = line.strip().split()
                             recipe1 = columns[0]
                             action_id = columns[1]
                             recipe2 = columns[2]
                             if len(columns)>4:
                                pred_label = columns[3][1]
                             else:
                                pred_label = columns[3]

                             
                             # dictionary with model predictions
                             # model_pred.keys() = recipe 1, action id
                             # model_pred.values() = recipe 2, model prediction                            
                             model_pred[(recipe1,action_id)] = (recipe2,pred_label)                             # print("pred",pred_label)
                             #print(model_pred)
                      else:
                         continue
                 
            print("gold", gold_pred)
            #print()
            #exit()
            print("model", model_pred)
               
            # sanity check
            if len(gold_pred)==len(model_pred):
               tot_pred = len(model_pred)
               print("yes they have the same lenght")
            else:
               print("len(gold_pred) is", len(gold_pred),"len(model_pred) is", len(model_pred))
               #print("Keys only in models' predictions:")
               #for key in model_pred.keys():
               #   if key not in gold_pred.keys():
                     #print(key)
               #print("Keys only in gold predictions:")
               #for key in gold_pred.keys():
               #   if key not in model_pred.keys():
                     #print(key)

            total_correct=0
            total=0

            # organize alignments' token ids in 2 lists: gold --> gold alignments, pred --> predicted by model
            for alignment in gold_pred.keys(): # ('pumpkin_chocolate_chip_bread_6', '5')
               total+=1
               #print("total", total)
               #print(alignment)
               #print(gold_pred[alignment],model_pred[alignment])

               # create a list of gold predictions in the order in which they are made
               if alignment in model_pred:
                  gold.append(gold_pred[alignment][1])
                  #print("gold",gold)

                  # create a list of model predictions in the order in which they are made
                  try:
                     pred.append(model_pred[alignment][1])
                     #print("pred", pred)
                     if gold_pred[alignment]==model_pred[alignment]:                   
                        total_correct+=1
                        #print("total_correct", total_correct)
                  except KeyError:
                     print(alignment)

            #diff = DeepDiff(gold_pred, model_pred)
            #print(len(diff['values_changed']))
            #print(alignment, gold_pred[alignment], model_pred[alignment])
            #print(total_correct, "correct predictions out of", len(gold_pred)) 
            #print(len(gold))
            #print(len(pred))

            # Accuracy
            #total_accuracy=0
            #total_accuracy=total_correct*100/tot_pred
            #print("The total accuracy is", total_accuracy)

            acc_score=accuracy_score(gold, pred) 
            #print(acc_score)

            # Precision & recall
            prec_recall_f1=precision_recall_fscore_support(gold , pred, average='weighted')
            #print(prec_recall_f1)
              
            # Save results in a dict
            source_recipe=rec1[:-2]
            results_per_dish[str(source_recipe)]={"accuracy":acc_score,"precision":prec_recall_f1[0],"recall":prec_recall_f1[1],"F1":prec_recall_f1[2]}

#print(results_per_dish)
print()

# Print stats related to results  
print("Dish-related performance statistics:")
# dish-related
for key in results_per_dish.keys():
   print("Results on dish",key,": Accuracy, Precision, Recall, F1:",results_per_dish[key]["accuracy"]*100,results_per_dish[key]["precision"]*100,results_per_dish[key]["recall"]*100,results_per_dish[key]["F1"]*100)
"""
# General
acc=[]
prec=[]
recall=[]
f1=[]
for key in results_per_dish.keys():
   acc.append(results_per_dish[key]["accuracy"])
   prec.append(results_per_dish[key]["precision"])
   recall.append(results_per_dish[key]["recall"])
   f1.append(results_per_dish[key]["F1"])  

print()
print("General performance statistics:")
print("Total accuracy:", sum(acc)/len(pred_files))
print("Total precision:", sum(prec)/len(pred_files))
print("Total recall:", sum(recall)/len(pred_files))
print("Total F1:", sum(recall)/len(pred_files))
"""           