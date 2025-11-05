import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score
from sklearn.metrics import confusion_matrix
import torch

metric_names = ['Recall','Precision','F1-Score','Specificity','PPV','NPV','AUC']
results_df = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
best_threshold = 0
metric_of_interest = 'Recall'
desired_metric_value = 0.85
error = 0.02



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return np.unravel_index(idx, array.shape)

def confusion_matrix(y_pred, y):
    true_pos = np.sum((y_pred == 1) & (y == 1))
    false_pos = np.sum((y_pred == 1) & (y == 0))
    true_neg = np.sum((y_pred == 0) & (y == 0))
    false_neg = np.sum((y_pred == 0) & (y == 1))
    return true_neg, false_neg, false_pos, true_pos

# def get_accuracy(y_pred, y):
#     return np.sum(1 for i in range(len(y)) if (y_pred[i].all() == y[i].all())) / len(y)

def get_threshold(yprob, ytrue, metric_of_interest, desired_metric_value, error):
    y = ytrue
    probs = yprob
    unique, frequency = np.unique(y,return_counts = True)
    threshold_metrics = pd.DataFrame(np.zeros((1000,7)),index=np.linspace(0,1,1000),columns=['Recall','Precision','F1-Score','Specificity','PPV','NPV','AUC'])
    prev = frequency[1]/len(y)
    for t in np.linspace(0,1,1000):
        pred = np.where(probs>t,1,0)
        true_neg, false_neg, false_pos, true_pos = confusion_matrix(pred, y)
        #accuracy = get_accuracy(pred, y)
        recall =  true_pos / (false_neg + true_pos)
        precision = true_pos / (false_pos + true_pos)
        specificity = true_neg/(true_neg+false_pos)
        #for ppv and npv, set prevalance 
        prev = frequency[1]/len(y)
        ppv = (recall* (prev))/(recall * prev + (1-specificity) * (1-prev))
        if true_neg== 0 and false_neg==0:
            npv = 0
        else:
            npv = (specificity* (1-prev))/(specificity * (1-prev) + (1-recall) * (prev))
        f1score = 2*(precision*recall)/(precision+recall)
        roc_auc = roc_auc_score(y, probs)
        threshold_metrics.loc[t,['Recall','Precision','F1-Score','Specificity','PPV','NPV','AUC']] = [recall,precision,f1score,specificity,ppv,npv,roc_auc]
    ### Find results for best threshold
    condition1 = threshold_metrics.loc[:,metric_of_interest] < (desired_metric_value + error)
    condition2 = threshold_metrics.loc[:,metric_of_interest] > (desired_metric_value - error)

    combined_condition = condition1 & condition2

    if metric_of_interest == 'Recall':
        sort_col = 'Precision'
    elif metric_of_interest == 'Precision':
        sort_col = 'Recall'
    elif metric_of_interest == 'F1-Score':
        sort_col = 'F1-Score'
    sorted_results = threshold_metrics[combined_condition].sort_values(by=sort_col, ascending=False)
    # print(sorted_results)
    if len(sorted_results) > 0:
        """ Only Record Value if Condition is Satisfied """
        results_df.loc[['Recall','Precision','F1-Score','Specificity','PPV','NPV','AUC'],1] = sorted_results.iloc[0,:]   
        best_threshold = sorted_results.iloc[0,:].name
    else:
        print('No Threshold Found for Constraint!')
    return best_threshold, results_df

        
def auroc_conf_intervals(y_true,y_pred):
    
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    n_bootstraps = 1000
    rng_seed = 25  # control reproducibility
    bootstrapped_scores = []
    bootstrapped_tpr = []
    bootstrapped_fpr = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        fpr, tpr, thresh = roc_curve(y_true[indices], y_pred[indices])
        #plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        bootstrapped_scores.append(score)
        bootstrapped_tpr.append(tpr)
        bootstrapped_fpr.append(fpr)
     
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    index_005 = bootstrapped_scores.index(confidence_lower)
    index_095 = bootstrapped_scores.index(confidence_upper)
   
    return(confidence_lower, confidence_upper,bootstrapped_fpr[index_005],bootstrapped_tpr[index_005],bootstrapped_fpr[index_095], bootstrapped_tpr[index_095],)  


def auprc_conf_intervals(y_true,y_pred):
    
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    n_bootstraps = 1000
    rng_seed = 25  # control reproducibility
    bootstrapped_scores = []
    bootstrapped_tpr = []
    bootstrapped_fpr = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = average_precision_score(y_true[indices], y_pred[indices])
        fpr, tpr, thresh = roc_curve(y_true[indices], y_pred[indices])
        #plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        bootstrapped_scores.append(score)
        bootstrapped_tpr.append(tpr)
        bootstrapped_fpr.append(fpr)
     
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    index_005 = bootstrapped_scores.index(confidence_lower)
    index_095 = bootstrapped_scores.index(confidence_upper)
   
    return(confidence_lower, confidence_upper,bootstrapped_fpr[index_005],bootstrapped_tpr[index_005],bootstrapped_fpr[index_095], bootstrapped_tpr[index_095],)

def sens_conf_intervals(y_true,y_pred,threshold):
    y_pred = np.where(y_pred>threshold,1,0)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    n_bootstraps = 1000
    rng_seed = 25  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = recall_score(y_true[indices], y_pred[indices])
        #plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        bootstrapped_scores.append(score)
     
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
   
    return(confidence_lower, confidence_upper)


def spec_conf_intervals(y_true,y_pred,threshold):
    y_pred = np.where(y_pred>threshold,1,0)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    n_bootstraps = 1000
    rng_seed = 25  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        
        ##tn, fp, fn, tp = confusion_matrix(y_true=y_true[indices], y_pred=y_pred[indices]).ravel()
        ##score = tn/(tn+fp)
        score = recall_score(y_pred=y_pred[indices], y_true=y_true[indices], pos_label=0)
        #plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        bootstrapped_scores.append(score)
     
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
   
    return(confidence_lower, confidence_upper)


def ppv_conf_intervals(y_true,y_pred,threshold):
    y_pred = np.where(y_pred>threshold,1,0)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    unique, frequency = np.unique(y_true,return_counts = True)
    
    n_bootstraps = 1000
    rng_seed = 25  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        
        #for ppv and npv, set prevalance 
        prev = frequency[1]/len(y_true)
        specificity = recall_score(y_pred=y_pred[indices], y_true=y_true[indices], pos_label=0)
        recall = recall_score(y_pred=y_pred[indices], y_true=y_true[indices])
        score = (recall* (prev))/(recall * prev + (1-specificity) * (1-prev))
        ##tn, fp, fn, tp = confusion_matrix(y_true=y_true[indices], y_pred=y_pred[indices]).ravel()
        ##score = tn/(tn+fp)
        
        #plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        bootstrapped_scores.append(score)
     
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
   
    return(confidence_lower, confidence_upper)


def npv_conf_intervals(y_true,y_pred,threshold):
    y_pred = np.where(y_pred>threshold,1,0)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    unique, frequency = np.unique(y_true,return_counts = True)
    
    n_bootstraps = 1000
    rng_seed = 25  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        
        #for ppv and npv, set prevalance 
        prev = frequency[1]/len(y_true)
        specificity = recall_score(y_pred=y_pred[indices], y_true=y_true[indices], pos_label=0)
        recall = recall_score(y_pred=y_pred[indices], y_true=y_true[indices])
        score = (specificity* (1-prev))/(specificity * (1-prev) + (1-recall) * (prev))
        ##tn, fp, fn, tp = confusion_matrix(y_true=y_true[indices], y_pred=y_pred[indices]).ravel()
        ##score = tn/(tn+fp)
        
        #plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        bootstrapped_scores.append(score)
     
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
   
    return(confidence_lower, confidence_upper)




def get_predictions(model,loader,loss_fn,device):
    P=[]
    L=[]
    model.eval()
    val_loss=0
    for i,batch in enumerate(loader):
        data,labels=batch
        data=data.to(torch.float32).to(device)
        labels=labels.to(torch.float32).to(device)
        
        pred=model(data)[:,0]
        loss=loss_fn(pred,labels)
        val_loss=val_loss+loss.item()

        P.append(pred.cpu().detach().numpy())
        L.append(labels.cpu().detach().numpy())
        
    val_loss=val_loss/len(loader)
    P=np.concatenate(P)  
    L=np.concatenate(L)
    return P,L



from Metrics import *

def get_scores(best_threshold,P,L):  
    P1 = np.where(P>best_threshold,1,0)

    sens_score = recall_score(y_pred=P1, y_true=L)
    spec_score = recall_score(y_pred=P1, y_true=L, pos_label=0)
    test_auc = roc_auc_score(y_score=P, y_true=L)
    test_aup = average_precision_score(y_score=P, y_true=L)


    unique, frequency = np.unique(L,return_counts = True)
    prev = frequency[1]/len(L)
    ppv_score = (sens_score* (prev))/(sens_score * prev + (1-spec_score) * (1-prev))
    npv_score = (spec_score* (1-prev))/(spec_score * (1-prev) + (1-sens_score) * (prev))
    
    
    print('AUROC:' + str(round(test_auc,3)) + "(" + str(round(auroc_conf_intervals(y_pred=P, y_true=L)[0],3)) + "-"+ str(round(auroc_conf_intervals(y_pred=P, y_true=L)[1],3)) + ")")
    print('AUPRC:' + str(round(test_aup,3)) + "(" + str(round(auprc_conf_intervals(y_pred=P, y_true=L)[0],3)) + "-"+ str(round(auprc_conf_intervals(y_pred=P, y_true=L)[1],3)) + ")")
    print('Sensitivity:' + str(round(sens_score,3)) + "(" + str(round(sens_conf_intervals(y_pred=P, y_true=L, threshold = best_threshold)[0],3)) + "-"+ str(round(sens_conf_intervals(y_pred=P, y_true=L, threshold = best_threshold)[1],3)) + ")")
    print('Specificity:' + str(round(spec_score,3)) + "(" + str(round(spec_conf_intervals(y_pred=P, y_true=L, threshold = best_threshold)[0],3)) + "-"+ str(round(spec_conf_intervals(y_pred=P, y_true=L, threshold = best_threshold)[1],3)) + ")")
    print('PPV:' + str(round(ppv_score,3)) + "(" + str(round(ppv_conf_intervals(y_pred=P, y_true=L, threshold = best_threshold)[0],3)) + "-"+ str(round(ppv_conf_intervals(y_pred=P, y_true=L, threshold = best_threshold)[1],3)) + ")")
    print('NPV:' + str(round(npv_score,3)) + "(" + str(round(npv_conf_intervals(y_pred=P, y_true=L, threshold = best_threshold)[0],3)) + "-"+ str(round(npv_conf_intervals(y_pred=P, y_true=L, threshold = best_threshold)[1],3)) + ")")
  
    
    return test_auc,test_aup 




def evaluate_model(model,loader, loss_fn,device):
    P,L=get_predictions(model,loader,loss_fn,device)
    metric_names = ['Recall','Precision','F1-Score','Specificity','PPV','NPV','AUC']
    results_df = pd.DataFrame(np.zeros((len(metric_names),1)),index=metric_names)
    best_threshold = 0
    metric_of_interest = 'Recall'
    desired_metric_value = 0.85
    error = 0.01
 
    best_threshold, results_df = get_threshold(P, L, metric_of_interest, desired_metric_value, error)
    
    P,L=get_predictions(model,loader,loss_fn,device)
    A,B=get_scores(best_threshold,P,L)

    