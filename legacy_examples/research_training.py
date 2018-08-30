import os
import sys
sys.path.append(os.getcwd())

from openrec import ImplicitModelTrainer
from openrec.utils import ImplicitDataset
from openrec.recommenders import FHMF, PMF, FLMF
from openrec.utils.evaluators import AUC, Recall, MSE
from openrec.utils.samplers import PointwiseSampler, PointwiseWSampler,PointwiseWSplitSampler, PointwiseAlphaSampler, PointwiseAlphaWSampler
from config import sess_config
from decimal import Decimal
import dataloader

import numpy as np

#Grid Search Params
dataset_name = sys.argv[1]
recommender = sys.argv[2]
l2_reg = float(sys.argv[3])
pos_ratio = float(sys.argv[4])
num_process= int(sys.argv[5])
dim_embed= int(sys.argv[6])
display_itr = int(sys.argv[7])
focus_bound = (int(sys.argv[8]), int(sys.argv[9]))
exp_factor = None
if recommender=="FHMF":
    exp_factor = float(sys.argv[10])
alpha = None
if recommender=="ALPHA_SAMPLER_PMF":
    alpha = float(sys.argv[10])
if recommender=="ALPHA_SAMPLER_FHMF":
    alpha = float(sys.argv[10])
    exp_factor = float(sys.argv[11])
if recommender=="FLMF":
    exp_factor = float(sys.argv[10])
    focused_i_reg = float(sys.argv[11])
    unfocused_i_reg = float(sys.argv[12])



#Training Params
batch_size = 1000
test_batch_size = 100
opt='Adam'
num_itr = 10001



def get_raw_data(dataset_name):
    raw_data=None
    if dataset_name == "amazon_book":
        raw_data = dataloader.load_amazon_book()
    elif dataset_name == "citeulike":
        raw_data = dataloader.load_citeulike()
    elif dataset_name == "tradesy":
        raw_data = dataloader.load_tradesy()
    else:
        print("Bad dataset name.")
        exit()
    return raw_data

def get_item_interaction_dict(full_dataset):
    interactions_per_item = {}
    for _, item_id in full_dataset:
        if item_id in interactions_per_item:
            interactions_per_item[item_id] += 1
        else:
            interactions_per_item[item_id] = 1
    return interactions_per_item

"""
def get_focus_bound(full_dataset, interactions_per_item):
    percent_focused = 0
    chosen_upper_bound = 0
    while percent_focused < percent_required:

        chosen_upper_bound += 1

        focus_bound = (1, chosen_upper_bound)

        num_focus_interactions = 0
        for _, item_id in full_dataset:
            if interactions_per_item[item_id] <= focus_bound[1] and interactions_per_item[item_id] >= focus_bound[0]:
                num_focus_interactions += 1

        percent_focused = int(100 * (float(num_focus_interactions) / float(np.size(full_dataset, 0))))

    return (1, chosen_upper_bound), percent_focused
"""

def get_focused(dataset, focus_bound, interactions_per_item):
    focused_indices = []
    for i, interaction in enumerate(dataset):
        user_id, item_id = interaction
        if interactions_per_item[item_id] < focus_bound[1] and interactions_per_item[item_id] >= focus_bound[0]:
            focused_indices.append(i)
    return np.take(dataset, focused_indices)


raw_data = get_raw_data(sys.argv[1])
val_dataset = ImplicitDataset(raw_data['val_data'], raw_data['max_user'], raw_data['max_item'], name='Val')
test_dataset = ImplicitDataset(raw_data['test_data'], raw_data['max_user'], raw_data['max_item'], name='Test')



#Getting Focused sets
full_dataset = np.concatenate((raw_data['train_data'], raw_data['val_data'], raw_data['test_data']), axis=0)
interactions_per_item = get_item_interaction_dict(full_dataset)
#focus_bound, percent_focused = get_focus_bound(full_dataset, interactions_per_item)
focused_val_set = get_focused(raw_data['val_data'], focus_bound, interactions_per_item)
focused_test_set = get_focused(raw_data['test_data'], focus_bound, interactions_per_item)
focused_dataset = np.concatenate((get_focused(raw_data['train_data'], focus_bound, interactions_per_item), focused_val_set, focused_test_set), axis=0)
percent_focused = int(100 * (float(len(focused_dataset) / float(np.size(full_dataset, 0)))))

train_dataset = ImplicitDataset(raw_data['train_data'], raw_data['max_user'], raw_data['max_item'], name='Train')
focused_val_set = ImplicitDataset(focused_val_set, raw_data['max_user'], raw_data['max_item'], name='Val')
focused_test_set = ImplicitDataset(focused_test_set, raw_data['max_user'], raw_data['max_item'], name='Test')


print("\r\n\r\nParams are:\r\n\r\n")
print("algorithm: %s" % (recommender))
print("dataset name: %s" % (dataset_name))
if recommender=="FLMF":
    print("l2_user_reg: %.1E" % Decimal(l2_reg))
else:
    print("l2_reg: %.1E" % Decimal(l2_reg))
print("pos_ratio: %.2f" % (pos_ratio))
print("num_process: %d" % (num_process))
print("dim_embed: %d" % (dim_embed))
print("display_itr: %d" % (display_itr))
print("focus bound: [%d,%d)" % (focus_bound[0], focus_bound[1]))
if recommender=="FHMF" or recommender=="ALPHA_SAMPLER_FHMF" or recommender=="FLMF" or recommender=="ALPHA_SAMPLER_FLMF":
    print("exp_factor: %.2f" % (exp_factor))
if recommender=="ALPHA_SAMPLER_PMF" or recommender=="ALPHA_SAMPLER_FHMF" or recommender=="ALPHA_SAMPLER_FLMF":
    print("alpha: %.2f" % (alpha))
if recommender=="FLMF" or recommender=="ALPHA_SAMPLER_FLMF":
    print("focused_i_reg: %.1E" % Decimal(focused_i_reg))
    print("unfocused_i_reg: %.1E" % Decimal(unfocused_i_reg))




print("The focus set composes %d%% of all the observations.\r\n" % (percent_focused))

num_negatives = None
if sys.argv[1] == "amazon_book":
    num_negatives = 500




if recommender=="FHMF":
    # Recommender
    fhmf_model = FHMF(batch_size=batch_size,
                      max_user=train_dataset.max_user(),
                      max_item=train_dataset.max_item(),
                      dim_embed=dim_embed,
                      opt='Adam',
                      l2_reg=l2_reg)
    #Sampler
    sampler = PointwiseWSampler(batch_size=batch_size, dataset=train_dataset, num_process=num_process, exp_factor=exp_factor,
                                         pos_ratio=pos_ratio)
    #Model Trainer
    eval_save_prefix='/home/ubuntu/openrec/models/fhmf_models/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}/'.format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10])
    if dataset_name=="amazon_book":
        eval_save_prefix=None
    model_trainer = ImplicitModelTrainer(batch_size=batch_size,
                                        test_batch_size=test_batch_size,
                                        train_dataset=train_dataset,
                                        model=fhmf_model,
                                        sampler=sampler,
                                         eval_save_prefix=eval_save_prefix)
    #Evaluators
    auc_evaluator = AUC()
    recall_evaluator = Recall(recall_at=[10, 50, 100, 300])
    #mse_evaluator = MSE()
    #Train
    model_trainer.train(num_itr=num_itr, display_itr=display_itr, eval_datasets=[focused_val_set, focused_test_set],
                        evaluators=[auc_evaluator, recall_evaluator], num_negatives=num_negatives)


elif recommender=="PMF":

    model = PMF(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(),
                    dim_embed=50, opt='Adam', sess_config=sess_config, l2_reg=l2_reg)
    sampler = PointwiseSampler(batch_size=batch_size, dataset=train_dataset, pos_ratio=pos_ratio, num_process=num_process)
    eval_save_prefix='/home/ubuntu/openrec/models/pmf_models/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}/'.format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9])
    if dataset_name=="amazon_book":
        eval_save_prefix=None
    model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size,
        train_dataset=train_dataset, model=model, sampler=sampler,
                                         eval_save_prefix=eval_save_prefix)

    auc_evaluator = AUC()
    recall_evaluator = Recall(recall_at=[10, 50, 100, 300])

    model_trainer.train(num_itr=num_itr, display_itr=display_itr, eval_datasets=[val_dataset, focused_test_set],
                        evaluators=[auc_evaluator, recall_evaluator], num_negatives=num_negatives)


elif recommender=="ALPHA_SAMPLER_PMF":



    model = PMF(batch_size=batch_size, max_user=train_dataset.max_user(), max_item=train_dataset.max_item(),
                    dim_embed=50, opt='Adam', sess_config=sess_config, l2_reg=l2_reg)
    
    # Sampler
    sampler = PointwiseAlphaSampler(dataset=train_dataset, batch_size=batch_size, pos_ratio=pos_ratio, num_process=num_process, alpha=alpha,
                                     focus_bound=focus_bound)  
    
    eval_save_prefix='/home/ubuntu/openrec/models/alpha_pmf_models/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}/'.format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10])
    if dataset_name=="amazon_book":
        eval_save_prefix=None
    model_trainer = ImplicitModelTrainer(batch_size=batch_size, test_batch_size=test_batch_size,
        train_dataset=train_dataset, model=model, sampler=sampler,
                                         eval_save_prefix=eval_save_prefix)

    auc_evaluator = AUC()
    recall_evaluator = Recall(recall_at=[10, 50, 100, 300])

    model_trainer.train(num_itr=num_itr, display_itr=display_itr, eval_datasets=[val_dataset, focused_test_set],
                        evaluators=[auc_evaluator, recall_evaluator], num_negatives=num_negatives)


elif recommender=="ALPHA_SAMPLER_FHMF":


    # Recommender
    fhmf_model = FHMF(batch_size=batch_size,
                      max_user=train_dataset.max_user(),
                      max_item=train_dataset.max_item(),
                      dim_embed=dim_embed,
                      opt='Adam',
                      l2_reg=l2_reg)
    #Sampler
    sampler = PointwiseAlphaWSampler(dataset=train_dataset, batch_size=batch_size, pos_ratio=pos_ratio, num_process=num_process, alpha=alpha,
                                     focus_bound=focus_bound, exp_factor=exp_factor) 
    #Model Trainer
    eval_save_prefix='/home/ubuntu/openrec/models/alpha_fhmf_models/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}/'.format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11])
    if dataset_name=="amazon_book":
        eval_save_prefix=None
    model_trainer = ImplicitModelTrainer(batch_size=batch_size,
                                        test_batch_size=test_batch_size,
                                        train_dataset=train_dataset,
                                        model=fhmf_model,
                                        sampler=sampler,
                                         eval_save_prefix=eval_save_prefix)
    #Evaluators
    auc_evaluator = AUC()
    recall_evaluator = Recall(recall_at=[10, 50, 100, 300])
    #Train
    model_trainer.train(num_itr=num_itr, display_itr=display_itr, eval_datasets=[focused_val_set, focused_test_set],
                        evaluators=[auc_evaluator, recall_evaluator], num_negatives=num_negatives)





if recommender=="FLMF":
    # Recommender
    flmf_model = FLMF(batch_size=batch_size,
                      max_user=train_dataset.max_user(),
                      max_item=train_dataset.max_item(),
                      dim_embed=dim_embed,
                      opt='Adam',
                      l2_reg_user=l2_reg,
                      l2_reg=l2_reg)
    
    #Sampler
    #sampler = PointwiseWSplitSampler(batch_size=batch_size, dataset=train_dataset, num_process=num_process, exp_factor=exp_factor,
    #                                     pos_ratio=pos_ratio, focused_reg=focused_i_reg, unfocused_reg=unfocused_i_reg, focus_bound=focus_bound)
    sampler = PointwiseWSampler(batch_size=batch_size, dataset=train_dataset, num_process=num_process, exp_factor=exp_factor,
                                         pos_ratio=pos_ratio)
    
    #Model Trainer
    eval_save_prefix='/home/ubuntu/openrec/models/flmf_models/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}/'.format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12])
    if dataset_name=="amazon_book":
        eval_save_prefix=None
    model_trainer = ImplicitModelTrainer(batch_size=batch_size,
                                        test_batch_size=test_batch_size,
                                        train_dataset=train_dataset,
                                        model=flmf_model,
                                        sampler=sampler,
                                         eval_save_prefix=eval_save_prefix)
    #Evaluators
    auc_evaluator = AUC()
    recall_evaluator = Recall(recall_at=[10, 50, 100, 300])
    #mse_evaluator = MSE()
    #Train
    model_trainer.train(num_itr=num_itr, display_itr=display_itr, eval_datasets=[focused_val_set, focused_test_set],
                        evaluators=[auc_evaluator, recall_evaluator], num_negatives=num_negatives)
                        

