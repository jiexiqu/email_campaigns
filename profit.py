import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def cost_benefit(tp=-3,fp=-3,fn=-9,tn=0):
    cost_benefit_matrix = np.array([[tp, fp], [fn,   tn]])
    return cost_benefit_matrix

def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])


def profit_curve(cost_benefit, predicted_probs, labels):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and thier true labels.
    Parameters
    ----------
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1
    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """
    n_obs = float(len(labels))
    # Make sure that 1 is going to be one of our thresholds
    # maybe_one = [] if 1 in predicted_probs else [1] 
    # thresholds = maybe_one + sorted(predicted_probs, reverse=True)

    thresholds = list(np.arange(0.05,1.05,0.05))
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / n_obs
        profits.append(threshold_profit)
        

    return np.array(profits), np.array(thresholds)

def profits_function(model, y_true, y_pred_probas, name_str):
    
    # create cost benefit matrix
    cbm = cost_benefit()

    # create profit curve
   
    profits, thresholds = profit_curve(cbm, y_pred_probas, y_true)

        # for profit, threshold in zip(*profits_curve_results)
    plt.plot(thresholds, profits, label = str(model.__class__.__name__) + name_str )
    
    plt.title("Profit Curves")
    plt.xlabel("Thresholds")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    
    
def find_best_threshold(model_profits):

    max_model = None
    max_threshold = None
    max_profit = None
    for model, profits, thresholds in model_profits:
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit


def find_max_threshold(y_true, y_pred_probas):
    cbm = cost_benefit()
    profits, thresholds = profit_curve(cbm, y_pred_probas, y_true)

    max_index = np.argmax(profits)
    
    max_threshold = thresholds[max_index]
    max_profit = profits[max_index]
    
    return max_profit, max_threshold
