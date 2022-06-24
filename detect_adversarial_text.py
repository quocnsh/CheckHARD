import matplotlib
matplotlib.use('Agg')
from sklearn.linear_model import LogisticRegression
import os
import torch
import argparse
import numpy as np
import textattack
import nltk
from pathlib import Path
import csv

from utils.load_model_command import load_model
from utils.attack_commands import load_attack
from utils.detect_commands import extract_features
from textattack.attack_results import SkippedAttackResult
from textattack.shared import  AttackedText
from textattack import Attacker
from textattack import AttackArgs
from textattack.datasets import Dataset

ADV_LABEL = 1
ORG_LABEL = 0

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
    parser.add_argument('--target_model',
                        help='Target model',
                        default="cnn-sst2")
    parser.add_argument('--auxiliary_attack',
                        help='An attack for defense',
                        default="pwws")
    parser.add_argument('--num_test',
                        help='Number test samples',
                        type=int,
                        default=10)
    parser.add_argument('--word_proportion',
                        help='word_proportion',
                        type=float,
                        default=1.0)
    parser.add_argument('--support_models',
                        nargs="*", 
                        help='List of support models.',
                       default = ['roberta-base-sst2'])
    parser.add_argument('--num_val',
                        help='Number dev sample',
                        type=int,
                        default=10)
    args = parser.parse_args()
    return args


def load_dataset_from_huggingface(dataset_name):
    dataset = textattack.datasets.HuggingFaceDataset(
        *dataset_name, shuffle=False
    )
    return dataset
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def init(attack):
    """ initilize the attack (required by TextAttack)
    Args:        
        attack:
            an attack from TextAttack
    Returns:        
    """ 
    original_text = AttackedText("this is a test")
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



def tokenize(text):
    return nltk.word_tokenize(text)

def read_data(file_name):
    with (open(file_name, 'rb')) as inp:
        org_texts, adv_texts, ground_truths = zip(*pickle.load(inp))
    return org_texts, adv_texts, ground_truths

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
            
    save_object(zip(org_texts,adv_texts, ground_truths), file_name)    



args = create_arguments()


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

class Sample:
    def __init__(self, text, detect_features, defense_features, predict_label, defense_label):
        self.text = text
        self.detect_features= detect_features
        self.defense_features = defense_features
        self.predict_label = predict_label
        self.defense_label = defense_label

# checked  
def save_all_data(org_data, adv_data, ground_truths, filename):
    save_object(zip(org_data,adv_data, ground_truths), filename)   

# checked
def load_all_data(file_name):
    with (open(file_name, 'rb')) as inp:
        org_data, adv_data, ground_truths = zip(*pickle.load(inp))
    return org_data, adv_data, ground_truths


# checked
def train_classifier(features, targets):
    features = np.array(features)
    clf = LogisticRegression(random_state=0).fit(features, targets)
#    clf = LinearRegression().fit(features, targets)
#    clf = MLPClassifier(random_state=1, max_iter=300).fit(features, targets)
    accuracy = clf.score(features, targets)
    # print_and_log(f"training accuracy = {accuracy}")
#    print(f"coef_ = {clf.coef_}")
    return clf


ADV_LABEL = 1
ORG_LABEL = 0
            
# checked    
def detect_adversarial(adv_classifier, sample, min_adv, max_adv, min_org, max_org):
    adv_detect = adv_classifier.predict([sample.detect_features])[0]
    max_rate = sample.detect_features[0]
    if (adv_detect == ADV_LABEL):
        if max_rate > max_adv[0]:
            max_adv[0] = max_rate
        if max_rate < min_adv[0]:
            min_adv[0] = max_rate
    else:
        if max_rate > max_org[0]:
            max_org[0] = max_rate
        if max_rate < min_org[0]:
            min_org[0] = max_rate
    return adv_detect
        
      
# checked    
def get_features_file_path(kind = "test"):
    supporters_name = '_'.join(args.support_models)
    assert kind == "test" or kind == "dev", "only support test or dev"
    if kind == "test":
        filename = f"test-{args.num_test}_supporters_{supporters_name}_attack_for_defense_{args.auxiliary_attack}_features.pkl"
    else:
        filename = f"dev-{args.num_val}_supporters_{supporters_name}_attack_for_defense_{args.auxiliary_attack}_features.pkl"
    filename = filename.replace('/', '_')
    return f'{args.dataset}/{args.target_model}/{args.original_attack}/{filename}'

# checked
def print_and_log(text):
    print(text)
    with open(f"{args.dataset}/{args.target_model}/{args.original_attack}/detect_adversarial_text_log.txt", "a+") as f:
        f.write(text + "\n")

# checked
def extract_data(texts, target_model, support_models, auxiliary_attack, batch_size = 64, word_proportion = 1.0):
    samples = []
    for index, text in enumerate(texts):
        sample = None
        if text != None:
            predict_label, defense_label, detect_features, defense_features = extract_features(text, target_model, support_models, auxiliary_attack, batch_size, word_proportion, handle_exception = True)
            sample = Sample(text, detect_features, defense_features, predict_label, defense_label)
        samples.append(sample)
        print(f"processed the sample {index + 1}")
    return samples           

# checked
def train_adv(org_data, adv_data, ground_truths):
    features = []
    targets = []
    for org_sample, adv_sample, ground_truth in zip(org_data, adv_data, ground_truths):
        features.append(org_sample.detect_features)
        targets.append(ORG_LABEL)
        if adv_sample != None and adv_sample.predict_label != ground_truth:
            features.append(adv_sample.detect_features)
            targets.append(ADV_LABEL)
    classifier = train_classifier(features, targets)
    return classifier    

# checked
def process(dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths):
    defense_predict_labels = []
    ori_predict_labels = []
    min_misclass = [2.0]
    min_correct = [2.0]
    max_misclass = [-1.0] 
    max_correct = [-1.0]
    
    
    adv_classifier = train_adv(dev_org_data, dev_adv_data, dev_ground_truths)
    adv_labels = []
    gold_labels = []
    min_adv = [2.0]
    min_org = [2.0]
    max_adv = [-1.0] 
    max_org = [-1.0]
    
    t_p = 0 # number of adv which is correctly detect
    f_p = 0 # number of org which is incorrectly detect
    pos = 0 # number of adv successfully attack target_model
    neg = 0 # number of original texts
    f_n = 0 # number of adv which is incorrectly detect    
   
    count = 0

    for org_sample, adv_sample, ground_truth in zip(test_org_data, test_adv_data, test_ground_truths):        
        adv_detect = detect_adversarial(adv_classifier, org_sample, min_adv, max_adv, min_org, max_org)
        count += 1
        print_and_log('\n' + '-' * 20 + f' SAMPLE {count} ' + '-' * 20 + '\n')
        print_and_log(f"Original : {org_sample.text}")
        print_and_log(f"Detection_result : {'Original (CORRECT)' if adv_detect ==  ORG_LABEL else 'Adversarial (INCORRECT)'}")

        gold_labels.append(ORG_LABEL)
        adv_labels.append(adv_detect)
        neg += 1
        if adv_detect != ORG_LABEL:
            f_p += 1
        if (adv_sample != None and adv_sample.predict_label != ground_truth):
            pos += 1            
            gold_labels.append(ADV_LABEL)
            adv_detect = detect_adversarial(adv_classifier, adv_sample, min_adv, max_adv, min_org, max_org)

            # count += 1
            # print_and_log('\n' + '-' * 20 + f' SAMPLE {count} ' + '-' * 20 + '\n')
            print_and_log(f"\nAdversarial : {adv_sample.text}")
            print_and_log(f"Detection_result : {'Adversarial (CORRECT)' if adv_detect ==  ADV_LABEL else 'Original (INCORRECT)'}")

            if adv_detect == ADV_LABEL:
                t_p += 1
            else:
                f_n += 1
            adv_labels.append(adv_detect)  
    tpr = t_p / pos if pos > 0 else 0
    fpr = f_p / neg if neg > 0 else 0
    f1 = (2 * t_p) / (2 * t_p + f_p + f_n) if 2 * t_p + f_p + f_n > 0 else 0
    return tpr, fpr, f1

# checked
def main():
    """ main processing
    """    
    test_feature_file = get_features_file_path(kind = "test")
    dev_feature_file = get_features_file_path(kind = "dev")
    if not os.path.exists(test_feature_file) or not os.path.exists(dev_feature_file):
        target_model = load_model(args.target_model)
        support_models = []
        original_attack = load_attack(args.original_attack, target_model)
        for supporter_name in args.support_models:
            support_models.append(load_model(supporter_name))
        auxiliary_attack = load_attack(args.auxiliary_attack, target_model)

    if not os.path.exists(f'{args.dataset}/{args.target_model}/{args.original_attack}'):
        os.makedirs(f'{args.dataset}/{args.target_model}/{args.original_attack}') 
    open(f"{args.dataset}/{args.target_model}/{args.original_attack}/detect_adversarial_text_log.txt", "w")
    
    if (args.num_val != 0):
        if not os.path.exists(f'{args.dataset}/{args.target_model}/{args.original_attack}/dev-{args.num_val}_data.pkl'):
            if (args.dataset == "sst2"):
                dataset = load_dataset_from_huggingface(TEST_SET_ARGS[args.dataset])
            else:
                dataset = load_dataset_from_huggingface(TRAIN_SET_ARGS[args.dataset])
            attack_args = AttackArgs(num_examples=args.num_val)
            attacker = Attacker(original_attack, dataset, attack_args)

            print('\n' + '*' * 30 + ' ATTACKING ON THE VALIDATION SET ' + '*' * 30 + '\n')
            attack_results = attacker.attack_dataset()
            save_data(dataset, attack_results, f'{args.dataset}/{args.target_model}/{args.original_attack}/dev-{args.num_val}_data.pkl')
        org_texts, adv_texts, ground_truths = read_data(f'{args.dataset}/{args.target_model}/{args.original_attack}/dev-{args.num_val}_data.pkl')        

        if not os.path.exists(dev_feature_file):
            print('\n' + '*' * 30 + ' OPTIMIZING MISSCLASSIFIED THRESHOLD ON THE VALIDATION SET ' + '*' * 30 + '\n')
            print('\n' +  'PROCESSING ORIGINAL TEXT '  + '\n')
            org_data = extract_data(org_texts, target_model, support_models, auxiliary_attack, batch_size = 32, word_proportion = args.word_proportion)
            print('\n' +  'PROCESSING ADVERSARIAL TEXT '  + '\n')
            adv_data = extract_data(adv_texts, target_model, support_models, auxiliary_attack, batch_size = 32, word_proportion = args.word_proportion)
            save_all_data(org_data, adv_data, ground_truths, dev_feature_file)

    if (args.num_test != 0):
        if not os.path.exists(f'{args.dataset}/{args.target_model}/{args.original_attack}/test-{args.num_test}_data.pkl'):
            if (args.dataset != "sst2"):
                dataset = load_dataset_from_huggingface(TEST_SET_ARGS[args.dataset])
            else:
                dataset = load_sst2_dataset()
            attack_args = AttackArgs(num_examples=args.num_test)
            attacker = Attacker(original_attack, dataset, attack_args)

            print('\n' + '*' * 30 + ' ATTACKING ON THE TESTING SET ' + '*' * 30 + '\n')
            attack_results = attacker.attack_dataset()
            save_data(dataset, attack_results, f'{args.dataset}/{args.target_model}/{args.original_attack}/test-{args.num_test}_data.pkl')
        org_texts, adv_texts, ground_truths = read_data(f'{args.dataset}/{args.target_model}/{args.original_attack}/test-{args.num_test}_data.pkl')        

        if not os.path.exists(test_feature_file):
            print('\n' + '*' * 30 + ' PROCESSING ON THE TESTING SET ' + '*' * 30 + '\n')
            print('\n' +  'PROCESSING ORIGINAL TEXT '  + '\n')
            org_data = extract_data(org_texts, target_model, support_models, auxiliary_attack, batch_size = 32, word_proportion = args.word_proportion)
            print('\n' +  'PROCESSING ADVERSARIAL TEXT '  + '\n')
            adv_data = extract_data(adv_texts, target_model, support_models, auxiliary_attack, batch_size = 32, word_proportion = args.word_proportion)
            save_all_data(org_data, adv_data, ground_truths, test_feature_file)
    dev_org_data, dev_adv_data, dev_ground_truths = load_all_data(dev_feature_file)
    test_org_data, test_adv_data, test_ground_truths = load_all_data(test_feature_file)
    print_and_log('\n' + '*' * 30 + ' DETECTING ADVERSARIAL TEXT ON TESTING SET ' + '*' * 30 + '\n')
    tpr, fpr, f1 = process(dev_org_data, dev_adv_data, dev_ground_truths, test_org_data, test_adv_data, test_ground_truths)

    print_and_log('\n' + '*' * 30 + ' SUMMARIZATION ' + '*' * 30 + '\n')
    print_and_log(f'EXPERIMENTIAL INFORMATION\n')
    print_and_log(f'Dataset : {args.dataset}')
    print_and_log(f'Number validation samples : {args.num_val}')        
    print_and_log(f'Number testing samples : {args.num_test}')        
    print_and_log(f'Original attack for generating adversarial text : {args.original_attack}')        
    print_and_log(f'Target model : {args.target_model}')
    print_and_log(f'Support models for CheckHARD : {args.support_models}')
    print_and_log(f'Auxiliary original_attack for CheckHARD : {args.auxiliary_attack}')
    print_and_log(f'word proportion : {args.word_proportion}')
    print_and_log(f'\nEVALUATION RESULTS\n')
    print_and_log(f'True Possitive Rate (TPR) : {tpr}')
    print_and_log(f'False Possitive Rate (FPR) : {fpr}')
    print_and_log(f'F1 : {f1}')


    print_and_log('*' * 80)

# checked
if __name__ == "__main__":
    main() 
    