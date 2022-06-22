import textattack
from textattack.attack_args import ATTACK_RECIPE_NAMES 
from textattack.shared import  AttackedText


def init(attack):
    """ initilize the attack (required by TextAttack)
    Args:        
        attack:
            an attack from TextAttack
    Returns:        
    """ 
    original_text = AttackedText("this is a test")
    initial_result,_ = attack.goal_function.init_attack_example(original_text, 0)

# checked    
def load_attack(attack_name, model_wrapper):
    """ Load attack from attack_name
    Args:
        attack_name (str):
            name of a attack, which are specified by TextAttack
    Returns:
        dataset (textattack.Attack):
            the attack corresponding with the dataset_name
    """ 

    if attack_name in ATTACK_RECIPE_NAMES:
        command = f"{ATTACK_RECIPE_NAMES[attack_name]}.build(model_wrapper)"
        recipe = eval(
                    command
                )
    else:
        raise ValueError(f"Error: unsupported recipe {attack_name}")
    recipe.cuda_() 
    init(recipe)
    return recipe