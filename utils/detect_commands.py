import math
import numpy as np
from textattack.shared import  AttackedText
import inspect
import torch

# checked
def predict(model, text):
    """ predict a text using a model
    Args:        
        model (textattack.models.wrappers.ModelWrapper):
            a model loaded by TextAttack
        text (str):
            a text
    Returns:      
        distribution (numpy array):
            distribution prediction for the input text
    """   
    distributions = model([text])
    if not(type(distributions[0]) is np.ndarray):
        return distributions[0].cpu().numpy()
    else:
        return distributions[0]

# checked
def get_candidate_indexes(pre_transformation_constraints, original_text, transformation):
    """ checking the input text (original text) with pre-constraints
    Args:
        pre_transformation_constraints: 
            list of pre-constrainst
        original_text:
            the original text
        transformation:
            transformation is used for checking
    Returns:
        indexes(array):
            list of indexes statified the pre-constrainst
    """
    indexes = None    
    for constrainst in pre_transformation_constraints:
        if indexes == None:
            indexes = set(constrainst(original_text, transformation))        
        else:
            indexes = indexes.intersection(constrainst(original_text, transformation))        
    return set(indexes)

# checked
def get_index_priority(search_method, initial_text, selected_indexes):
    """ sort the selected_indexes by priority
    Args:
        search_method:
            the method determines the priority
        initial_text:
            the original text
        selected_indexes:
            indexes are statisfied the pre-constrainsts
        
    Returns:
        indexes(array):
            list of indexes after sorting
    """    
    if (hasattr(search_method, '_get_index_order') and inspect.isfunction(search_method._get_index_order)):
        index_order, search_over = search_method._get_index_order(initial_text)
        indexes = []
        for index in index_order:
            if index in selected_indexes:
               indexes.append(index) 
        return indexes
    else:
        return selected_indexes

# checked
def pre_constrainst(attack, input_text):
    """ determine the priority for defense
    Args:
        attack (textattack.Attack):
            an attack from TextAttack
        input_text:
            the original text
    Returns:
        indexes(array):
            list of indexes after priority
    """    
    original_text = AttackedText(input_text)
    indexes = get_candidate_indexes(attack.pre_transformation_constraints, original_text, attack.transformation)  
    return list(indexes)
    
# checked
def priority(attack, input_text):
    """ determine the priority for defense
    Args:
        attack (textattack.Attack):
            an attack from TextAttack
        input_text:
            the original text
    Returns:
        indexes(array):
            list of indexes after priority
    """    
    original_text = AttackedText(input_text)
    return priority_by_attack(attack, input_text)

# checked
def priority_by_attack(attack, input_text):
    """ determine the priority for defense
    Args:
        attack (textattack.Attack):
            an attack from TextAttack
        input_text:
            the original text
    Returns:
        indexes(array):
            list of indexes after priority
    """    
    original_text = AttackedText(input_text)
    indexes = get_candidate_indexes(attack.pre_transformation_constraints, original_text, attack.transformation)  
    indexes = get_index_priority(attack.search_method, original_text, list(indexes))
    return indexes



# checked
def transform(transformation, input_text, word_index):
    """ transform a word at the word_index in input_text
    Args:
        transformation:
            transformation of an attack from TextAttack
        input_text:
            the original text
        word_index:
            index of word to modify
    Returns:
        transform_texts(array):
            list of texts after transformation
    """ 
    original_text = AttackedText(input_text)
    transform_texts = transformation(original_text, indices_to_modify = [word_index])
    return transform_texts

# checked
def constraint(transformed_texts, attack, input_text, handle_exception = False):
    """ check constrainsts for transformed texts
    Args:        
        transformed_texts:
            transformed texts
        attack:
            An attack from TextAttack framework
        input_text:
            an input text
    Returns:                
        statisfied_texts: 
            all texts in the transformed texts statify the constrainsts
    """ 
    result = []
    if handle_exception:
        try:
            original_text = AttackedText(input_text)
            filtered_texts = attack.filter_transformations(transformed_texts, original_text, original_text)            
            for filter_text in filtered_texts:
                result.append(filter_text.text.strip())
            return result
        except:
            return result
    else:
        original_text = AttackedText(input_text)
        filtered_texts = attack.filter_transformations(transformed_texts, original_text, original_text)            
        for filter_text in filtered_texts:
            result.append(filter_text.text.strip())
        return result
        
    
# checked
def batch_model_predict(model_predict, inputs, batch_size=64):
    """Runs prediction on iterable ``inputs`` using batch size ``batch_size``.

    Aggregates all predictions into an ``np.ndarray``.
    """
    outputs = []
    i = 0
    while i < len(inputs):
        batch = inputs[i : i + batch_size]
        batch_preds = model_predict(batch)

        # Some seq-to-seq models will return a single string as a prediction
        # for a single-string list. Wrap these in a list.
        if isinstance(batch_preds, str):
            batch_preds = [batch_preds]

        # Get PyTorch tensors off of other devices.
        if isinstance(batch_preds, torch.Tensor):
            batch_preds = batch_preds.cpu()

        # Cast all predictions iterables to ``np.ndarray`` types.
        if not isinstance(batch_preds, np.ndarray):
            batch_preds = np.array(batch_preds)
        outputs.append(batch_preds)
        i += batch_size

    return np.concatenate(outputs, axis=0)

# checked
def detect_perturbed_words(input_text, target, attack_for_defense, batch_size = 64, word_ratio = 1.0, verbose = True, handle_exception = False, ignore_word_importance = True):
    victim_distribution = predict(target, input_text)
    predict_index = np.argmax(victim_distribution)     
    victim_confidence = victim_distribution[predict_index]
    original_text = AttackedText(input_text)
    # word_indexes = get_candidate_indexes(attack_for_defense.pre_transformation_constraints, original_text, attack_for_defense.transformation)  # checked

    if ignore_word_importance:
        word_indexes = pre_constrainst(attack_for_defense, input_text) # checked
    else:
        word_indexes = priority(attack_for_defense, input_text) # checked
    if ignore_word_importance:
        np.random.seed(0)
        np.random.shuffle(word_indexes)
        
    num_word = int(math.floor(len(word_indexes) * word_ratio))
    word_indexes = word_indexes[:num_word]    

    # word_indexes = sorted(list(word_indexes))
    distances = []
    count_differences = []
    ratios_differences = []
    words = []
    indexes = []
    for word_index in word_indexes:
        texts = []
        transformed_texts = transform(attack_for_defense.transformation, input_text, word_index) # checked
        filtered_texts = constraint(transformed_texts, attack_for_defense, input_text, handle_exception) # checked
        for transformed_text in filtered_texts:
            texts.append(transformed_text) 
        distance = -999999
        count = 0
        if (len(texts) > 0):
            merge_distribution = batch_model_predict(target, texts, batch_size = batch_size) # checked
            for distribution in merge_distribution:
                if (np.argmax(distribution) != predict_index):
                    count += 1
                if (victim_confidence - distribution[predict_index] > distance):                    
                    distance = victim_confidence - distribution[predict_index]
            distances.append(distance)
            count_differences.append(count)
            ratios_differences.append(count/(len(texts)))
            indexes.append(word_index)
            words.append(original_text.words[word_index])
    n = len(words)
    sorted_indexes = [i for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if ratios_differences[sorted_indexes[i]] < ratios_differences[sorted_indexes[j]]:
                temp = sorted_indexes[i]
                sorted_indexes[i] = sorted_indexes[j]
                sorted_indexes[j] = temp
    suggest_words = []
    for i in range(n):
        duplicate = False
        for word in suggest_words:
            if words[sorted_indexes[i]] == word:
                duplicate = True
                break
        if not duplicate:
            suggest_words.append(words[sorted_indexes[i]])
    return suggest_words

# checked
def extract_features(input_text, target, supporters, attack_for_defense, batch_size, word_ratio = 1.0, handle_exception = False, ignore_word_importance = True):
    detect_features = []
    defense_features = []
    victim_distribution = predict(target, input_text) # checked
    predict_label = np.argmax(victim_distribution)     
    defense_label = 1 - predict_label
    
    if ignore_word_importance:
        word_indexes = pre_constrainst(attack_for_defense, input_text) # checked
    else:
        word_indexes = priority(attack_for_defense, input_text) # checked
    if ignore_word_importance:
        np.random.seed(0)
        np.random.shuffle(word_indexes)
    num_word = int(math.floor(len(word_indexes) * word_ratio))
    word_indexes = word_indexes[:num_word]    
    
    max_rate = 0
    for word_index in word_indexes:
        texts = []
        transformed_texts = transform(attack_for_defense.transformation, input_text, word_index) # checked
        filtered_texts = constraint(transformed_texts, attack_for_defense, input_text, handle_exception) # checked
        for transformed_text in filtered_texts:
            texts.append(transformed_text) 
        
        count = 0
        count_all = 0
        if (len(texts) > 0):
            merge_distribution = batch_model_predict(target, texts, batch_size = batch_size) # checked
            for distribution in merge_distribution:
                count_all += 1
                if (np.argmax(distribution) != predict_label):
                    count += 1
            for supporter in supporters:
                support_distribution = batch_model_predict(supporter, texts, batch_size = batch_size) # checked
                for sup_distribution in support_distribution:
                    count_all += 1
                    if (np.argmax(sup_distribution) != predict_label):
                        count += 1
            if count / count_all > max_rate:
                max_rate = count / count_all
    detect_features.append(max_rate)
    defense_features.append(max_rate)
    return predict_label, defense_label, detect_features, defense_features