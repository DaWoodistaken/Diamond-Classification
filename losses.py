import numpy as np

def loss(probs, y):
    '''
        Calculate the softmax loss
        --------------------------
        :param probs: softmax probabilities
        :param y: correct labels
        :return: loss
    '''
    loss = None

    #### Your implementation starts ######

    # compute the loss
    N = probs.shape[0]  # Number of training data

    # Compute the cross-entropy loss
    correct_log_probs = -np.log(probs[np.arange(N), y] + 1e-12)  # Adding epsilon for numerical stability
    loss = np.sum(correct_log_probs) / N  # Average loss

    ##### End of your implementation #####
    return loss

