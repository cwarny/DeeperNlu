import numpy as np
import json


def get_nbest_hypotheses(slot_scores, intent_scores, length, intent_ids, slot_ids, k=300):
    if slot_scores.shape[1] > intent_scores.shape[0]:
        intent_slot_probs = np.vstack([
            np.pad(intent_scores, (0, slot_scores.shape[1] - intent_scores.shape[0]), 'constant', constant_values=(0, 0)), 
            slot_scores
        ])
    elif slot_scores.shape[1] < intent_scores.shape[0]:
        intent_slot_probs = np.vstack([
            intent_scores, 
            np.pad(slot_scores, ((0,0), (0, intent_scores.shape[0] - slot_scores.shape[1])), 'constant', constant_values=(0, 0))
        ])
    nbest = get_nbest(intent_slot_probs, nbest_num=k, len_mask=length+1)
    hypotheses = []
    for option, confidence in nbest:
        domain_intent, *labels = option
        domain, intent = intent_ids.get_word(domain_intent).split('_')
        labels = [slot_ids.get_word(idx) for idx in labels][:length]
        hypotheses.append({
            'domain': domain,
            'intent': intent,
            'labels': labels,
            'confidence': float(confidence)
        })
    return hypotheses


def get_nbest(probability_mat, nbest_num=5, len_mask=None):
    '''
    See Nbest computation section for details.  This function takes a matrix of hypothesized probabilities and returns a list of lists of indices where each top level list is an nbest hypothesis, in sorted order.
    Input:
    probability_mat: num_timestepsXnum_possible_hyp_per_timestep (e.g. num labels)
    nbest_num: number of nbest to search
    len_mask: length mask (int) after which we shouldn't consider probabilities anymore because it's all zero buffer. (e.g. input utt length)
    Output:
    real_indices:  list of lists where each sublist an nbest element and is of form [list_of_indices,probability]
    '''

    if len_mask:
        probability_mat = probability_mat[:len_mask, :]
    # create a new matrix with indices in the sorted order then use it to sort the actual probabilities (to save time)
    probability_mat_indices = np.argsort(-probability_mat)
    probability_mat_sorted = np.take_along_axis(probability_mat, probability_mat_indices, axis=1)
    # generate the number of timesteps and labels from the size of the matrix
    num_timesteps, num_labels = probability_mat.shape
    # initialize the nbest and next_nbest as the first column (always highest probability)
    next_nbest = (np.zeros(num_timesteps, dtype=int),
                  np.prod(probability_mat_sorted[range(num_timesteps), np.zeros(num_timesteps, dtype=int)]))
    nbest = [next_nbest]
    all_seen_states = set()
    # initialize the stack as empty before we start adding things to it
    stack = np.empty((0, num_timesteps), dtype=int)
    stack_prob = np.empty((0, 1))
    for i in range(1, nbest_num):
        # add everything accessible  from selected nbest[-1] to the stack
        # (technically there are cases where we don't need to check all, maybe keep list of lowest visit for each then don't go if lower than all)
        # (i.e. expand list of nodes by adding next increments to stack)
        # (e.g. if our last nbest was [1,2,1] we will add the following to the stack:
        # [2,2,1],[1,3,1],[1,2,2])
        for time_step in range(num_timesteps):
            if nbest[-1][0][time_step] < num_labels:
                new_state_idx_list = np.array(nbest[-1][0], dtype=int)
                new_state_idx_list[time_step] = new_state_idx_list[time_step] + 1
                tmp = tuple(new_state_idx_list)
                #if we haven't seen the state being expanded, append it to the stack for future eval
                if tmp not in all_seen_states:
                    stack = np.append(stack, [new_state_idx_list], axis=0)
                    stack_prob = np.append(stack_prob, np.prod(probability_mat_sorted[range(num_timesteps), new_state_idx_list]))
                    all_seen_states.add(tmp)
        #check to make sure the stack isn't empty
        if stack.shape[0] == 0:
            print("Exhausted all nbest")
            return nbest
        #find the best element in the stack to expand and add to the nbest
        next_nbest_idx = np.argmax(stack_prob)
        #add it to the nbest and remove it from the stack
        nbest.append([stack[next_nbest_idx, :], stack_prob[next_nbest_idx]])
        stack = np.delete(stack, next_nbest_idx, 0)
        stack_prob = np.delete(stack_prob, next_nbest_idx, 0)
    real_indices = []

    #normalize probabilities and convert to the actual indices, rather than the ones relative to the sorted matrix
    normalized_probs = np.array([best[1] for best in nbest]) / np.sum(np.array([best[1] for best in nbest]))
    for i, best in enumerate(nbest):
        real_indices.append([probability_mat_indices[range(num_timesteps), best[0]], normalized_probs[i]])
    return real_indices


