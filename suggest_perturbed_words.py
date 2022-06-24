import matplotlib
matplotlib.use('Agg')
import os
import torch
import argparse
import numpy as np # checked
import textattack
import nltk # checked
from pathlib import Path
import csv # checked

try:
    import cPickle as pickle # checked

except ModuleNotFoundError:
    import pickle # checked


from utils.load_model_command import load_model # unchecked
from utils.attack_commands import load_attack # unchecked load_attack
from utils.detect_commands import detect_perturbed_words # checked detect_perturbed_words
from textattack.attack_results import SkippedAttackResult # checked SkippedAttackResult
from textattack.shared import  AttackedText # checked AttackedText
from textattack import Attacker # checked
from textattack import AttackArgs # checked

TRAIN_SET_ARGS = {
        "sst2": ("glue","sst2","train"),
        "imdb": ("imdb",None,"train"),
        "ag_news": ("ag_news",None,"train"),
        "cola": ("glue", "cola", "train"),
        "mrpc": ("glue", "mrpc", "train"),
        "qnli": ("glue", "qnli", "train"),
        "rte": ("glue", "rte", "train"),
        "wnli": ("glue", "wnli", "train"),
        "mr": ("rotten_tomatoes", None, "train"),
        "snli": ("snli", None, "train"),
        "yelp": ("yelp_polarity", None, "train"),
        }
TEST_SET_ARGS = {
        "sst2": ("glue","sst2","validation"),
        "imdb": ("imdb",None,"test"),
        "ag_news": ("ag_news",None,"test"),
        "cola": ("glue", "cola", "validation"),
        "mrpc": ("glue", "mrpc", "validation"),
        "qnli": ("glue", "qnli", "validation"),
        "rte": ("glue", "rte", "validation"),
        "wnli": ("glue", "wnli", "validation"),
        "mr": ("rotten_tomatoes", None, "test"),
        "snli": ("snli", None, "test"),
        "yelp": ("yelp_polarity", None, "test"),
        }

# checked
def create_arguments():
    """ Create all arguments
    Args:        
    Returns:      
        args (argparse.ArgumentParser):
            all aguments for CheckHARD

    """
    parser = argparse.ArgumentParser(description='CheckHARD')
    parser.add_argument('--dataset',
                        help='A dataset',
                        default="sst2")                        
    parser.add_argument('--original_attack',
                        help='An original_attack',
                        default="pwws")
    parser.add_argument('--auxiliary_attack',
                        help='An attack for defense',
                        default="pwws")
    parser.add_argument('--target_model',
                        help='Target model',
                        default="cnn-sst2")
    # parser.add_argument('--supporters',
    #                     nargs="*", 
    #                     help='List of support models.',
    #                     default = [])
    parser.add_argument('--num_test',
                        help='Number test samples',
                        type=int,
                        default=10)
    parser.add_argument('--suggestion_number',
                        help='suggestion number',
                        type=int,
                        default=3)
    parser.add_argument('--word_proportion',
                        help='word_proportion',
                        type=float,
                        default=1.0)
    args = parser.parse_args()
    return args


# checked
def load_dataset_from_huggingface(dataset_name):
    dataset = textattack.datasets.HuggingFaceDataset(
        *dataset_name, shuffle=False
    )
    return dataset

# checked
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# checked
def init(attack):
    """ initilize the attack (required by TextAttack)
    Args:        
        attack:
            an attack from TextAttack
    Returns:        
    """ 
    original_text = AttackedText("this is a test") # checked
    initial_result,_ = attack.goal_function.init_attack_example(original_text, 0)


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
    return batch_model_predict(model, [text])[0]
    distributions = model([text])
    if not(type(distributions[0]) is np.ndarray):
        return distributions[0].cpu().numpy()
    else:
        return distributions[0]

# checked
def tokenize(text):
    return nltk.word_tokenize(text)

# checked
def read_data(file_name):
    with (open(file_name, 'rb')) as inp:
        org_texts, adv_texts, ground_truths = zip(*pickle.load(inp))
    return org_texts, adv_texts, ground_truths

# checked
def save_data(dataset, attack_results, file_name):
    org_texts = []
    adv_texts = []
    ground_truths = []
    for i, result in enumerate(attack_results):
        if isinstance(result, SkippedAttackResult):
            __, ground_truth = dataset[i]
            org_text = result.original_text()
            adv_text = None
            org_texts.append(org_text)
            adv_texts.append(adv_text)
            ground_truths.append(ground_truth)
        else:
            __, ground_truth = dataset[i]
            org_text = result.original_text()
            adv_text = result.perturbed_text()
            org_texts.append(org_text)
            adv_texts.append(adv_text)
            ground_truths.append(ground_truth)
            
    save_object(zip(org_texts,adv_texts, ground_truths), file_name)    # checked


# checked
def get_perturbed_words(org_text, adv_text):
    original_words = tokenize(org_text)
    adversarial_words = tokenize(adv_text)
    perturbed_words, indexes = [], []
    for index in range(min(len(original_words), len(adversarial_words))):
        if (original_words[index] != adversarial_words[index]):
            perturbed_words.append(adversarial_words[index])
            indexes.append(index)
    return perturbed_words, indexes
    
# checked
def evaluate_word_detection(suggest_words, perturbed_words, adv_text):
    adv_words = tokenize(adv_text)
    perturbed_words = set(perturbed_words)
    for index, suggest_word in enumerate(suggest_words):
        if suggest_word in perturbed_words:
            return index + 1
    return len(adv_words)
      

args = create_arguments()

# checked
def load_sst2_dataset(file_name = "data/sst2/test.tsv"):
    tsv_file = open(file_name)
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    data = []
    is_header = True
    for row in read_tsv:
        if is_header:
            is_header = False
        else:
            data.append((row[0], int(row[1])))
    tsv_file.close()
    dataset = textattack.datasets.Dataset(data)
    return dataset

# checked
def print_and_log(text):
    print(text)
    with open(f"{args.dataset}/{args.target_model}/{args.original_attack}/detect_perturbed_words_log.txt", "a+") as f:
        f.write(text + "\n")

# checked
def evaluate_word_detection_array(word_detection, top = 1):
    count = 0
    for detection in word_detection:
        if detection <= top:
            count+= 1
    accuracy = count / len(word_detection)
    return accuracy, count

# checked
# def suggest_preturbed_words(org_texts, adv_texts, ground_truths, target_model, supporters, auxiliary_attack, batch_size = 32, word_proportion = 1.0, suggestion_number = 3):    
def suggest_preturbed_words(org_texts, adv_texts, ground_truths, target_model, auxiliary_attack, batch_size = 32, word_proportion = 1.0, suggestion_number = 3):    
    word_detection = []
    count = 0
    for i, adv_text in enumerate(adv_texts):    
        if adv_text != None:
            ori_predict = np.argmax(predict(target_model, org_texts[i]))
            adv_predict = np.argmax(predict(target_model, adv_texts[i]))
            if (ori_predict == ground_truths[i] and adv_predict != ground_truths[i]): # successful attack
                count += 1
                print_and_log('\n' + '-' * 20 + f' SAMPLE {count} ' + '-' * 20 + '\n')
                print_and_log(f"Original text : {org_texts[i]}")
                print_and_log(f"Adversarial text : {adv_texts[i]}")
                perturbed_words, indexes = get_perturbed_words(org_texts[i], adv_texts[i]) # checked
                print_and_log(f"\nActually preturbed words : {perturbed_words}")
                suggest_words= detect_perturbed_words(adv_text, target_model, auxiliary_attack, word_ratio = word_proportion,  batch_size = batch_size, verbose = True) # checked
                print_and_log(f"k suggested words : {suggest_words[:suggestion_number]}")
                lowest_correction = evaluate_word_detection(suggest_words, perturbed_words, adv_texts[i]) # checked
                word_detection.append(lowest_correction)
                if lowest_correction <= suggestion_number:
                    print(f"Matching result : CORRECT")
                else:
                    print(f"Matching result : INCORRECT")
                
    total_adv = len(word_detection)
    accuracy, matching_correct = evaluate_word_detection_array(word_detection, top = suggestion_number)   # checked     
    return accuracy, matching_correct, total_adv

# checked
def main():
    """ main processing
    """    
    
    target_model = load_model(args.target_model) # checked
    # supporters = []
    # for supporter_name in args.supporters:
    #     supporters.append(load_model(supporter_name))
    original_attack = load_attack(args.original_attack, target_model) # checked
    init(original_attack) # checked
    auxiliary_attack = load_attack(args.auxiliary_attack, target_model)
    init(auxiliary_attack)
    original_attack.cuda_()
    auxiliary_attack.cuda_()

    if not os.path.exists(f'{args.dataset}/{args.target_model}/{args.original_attack}'):
        os.makedirs(f'{args.dataset}/{args.target_model}/{args.original_attack}')            
        
    open(f"{args.dataset}/{args.target_model}/{args.original_attack}/detect_perturbed_words_log.txt", "w")

    if (args.num_test != 0):
        if not os.path.exists(f'{args.dataset}/{args.target_model}/{args.original_attack}/test-{args.num_test}_adv_data.pkl'):
            # start=datetime.now()
            if (args.dataset != "sst2"):
                dataset = load_dataset_from_huggingface(TEST_SET_ARGS[args.dataset]) # checked
            else:
                dataset = load_sst2_dataset() # checked
            attack_args = AttackArgs(num_successful_examples=args.num_test)
            attacker = Attacker(original_attack, dataset, attack_args)
            print("Attacking on testing set")
            attack_results = attacker.attack_dataset()
            save_data(dataset, attack_results, f'{args.dataset}/{args.target_model}/{args.original_attack}/test-{args.num_test}_adv_data.pkl') # checked
        org_texts, adv_texts, ground_truths = read_data(f'{args.dataset}/{args.target_model}/{args.original_attack}/test-{args.num_test}_adv_data.pkl') # checked        

        print_and_log('\n' + '*' * 30 + ' PERTURBED WORD SUGGESTION ' + '*' * 30 + '\n') # checked
        # accuracy, matching_correct, total_adv = suggest_preturbed_words(org_texts, adv_texts, ground_truths, target_model, supporters, auxiliary_attack, batch_size = 32, word_proportion = args.word_proportion, suggestion_number = args.suggestion_number) # checked
        accuracy, matching_correct, total_adv = suggest_preturbed_words(org_texts, adv_texts, ground_truths, target_model,  auxiliary_attack, batch_size = 32, word_proportion = args.word_proportion, suggestion_number = args.suggestion_number) # checked
        print_and_log('\n' + '*' * 30 + ' SUMMARIZATION ' + '*' * 30 + '\n')
        print_and_log(f'EXPERIMENTIAL INFORMATION\n')
        print_and_log(f'Dataset : {args.dataset}')
        print_and_log(f'Number testing samples : {args.num_test}')        
        print_and_log(f'Original attack for generating adversarial text : {args.original_attack}')        
        print_and_log(f'Target model : {args.target_model}')
        print_and_log(f'Auxiliary original_attack for CheckHARD : {args.auxiliary_attack}')
        print_and_log(f'Suggestion number : {args.suggestion_number}')
        print_and_log(f'\nEVALUATION RESULTS\n')
        print_and_log(f'Matching correct : {matching_correct} of {total_adv}')
        print_and_log(f'Top-{args.suggestion_number} accuracy : {accuracy}')
    print_and_log('*' * 70)

# checked
if __name__ == "__main__":
    main() 