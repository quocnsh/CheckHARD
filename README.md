# CheckHARD: Checking Hard Labels for Adversarial Text Detection, Prediction Correction, and Perturbed Word Suggestion

## Dependencies

* TextAttack framework (https://github.com/QData/TextAttack) (install by : `pip install textattack`)

## Usage

* Run the following commmands for corresponding objectives:
 `python3 detect_adversarial_text.py`
 `python3 correction_prediction.py`
 `python3 suggest_preturbed_words.py`


### Parameters

#### Parameters for all three objectives:

* `dataset` : dataset name (default = 'sst2'), complied with the names from HuggingFaceDataset (https://github.com/huggingface/datasets/tree/master/datasets) such as 'imdb', 'mrpc', 'yelp.'
* `original_attack`: attack is used for generating adversarial text (default = 'pwws'), compied with the names from TextAttack. Other attack names can be found in the TextAttack framework (https://textattack.readthedocs.io/en/latest/3recipes/attack_recipes_cmd.html#attacks-and-papers-implemented-attack-recipes-textattack-attack-recipe-recipe-name)
* `target_model`: name of target model (default = 'cnn-ag-news'), complied with the names from TextAttack. Other model name can be found in the TextAttack framework (https://textattack.readthedocs.io/en/latest/3recipes/models.html#textattack-models)
* `auxiliary_attack`: an attack is used for CheckHARD (default = 'pwws')
* `num_test`: number of testing samples (default = 10)
* `word_proportion`: the proportion of processing words  (default = 1.0)

#### Parameters for Adversarial Text Detection and Prediction Correction:

* `support_models`: list of support-model names (default = ['roberta-base-sst2'])
* `num_val`:  number of validation samples (default = 10)

#### Parameter for Perturbed Word Suggestion:

* `suggestion_number`: number of suggested words (default = 3)

### Examples

* Running with default parameters : `python3 detect_adversarial_text.py`
* Running with customized parameters : `python3 detect_adversarial_text.py --dataset imdb --original_attack textfooler --target_model cnn-imdb --auxidiary_attack deepwordbug --num_test 5 --word_proportion 0.9 --support_models lstm-imdb bert-base-uncased-imdb --num_val 15`

### Acknowledgement
* TextAttack framework (https://github.com/QData/TextAttack)

