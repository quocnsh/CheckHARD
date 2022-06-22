import transformers

import textattack
from textattack.model_args import HUGGINGFACE_MODELS, TEXTATTACK_MODELS

# checked
def load_model(model_name):
    """ Load model from model_name
    Args:
        model_name (str):
            name of a model, which are specified by TextAttack
    Returns:
        dataset (textattack.models.wrappers.ModelWrapper):
            the wrapper model corresponding with the model_name
    """         
    if model_name in TEXTATTACK_MODELS:
        # Support loading TextAttack pre-trained models via just a keyword.
        colored_model_name = textattack.shared.utils.color_text(
            model_name, color="blue", method="ansi"
        )
        if model_name.startswith("lstm"):
            textattack.shared.logger.info(
                f"Loading pre-trained TextAttack LSTM: {colored_model_name}"
            )
            model = textattack.models.helpers.LSTMForClassification.from_pretrained(
                model_name
            )
            model.cuda()
        elif model_name.startswith("cnn"):
            textattack.shared.logger.info(
                f"Loading pre-trained TextAttack CNN: {colored_model_name}"
            )
            model = (
                textattack.models.helpers.WordCNNForClassification.from_pretrained(
                    model_name
                )
            )           
            model.cuda()
        elif model_name.startswith("t5"):
            model = textattack.models.helpers.T5ForTextToText.from_pretrained(
                model_name
            )
            model.cuda()
        else:
            raise ValueError(f"Unknown textattack model {model_name}")

        # Choose the approprate model wrapper (based on whether or not this is a HuggingFace model).
        if isinstance(model, textattack.models.helpers.T5ForTextToText):
            model = textattack.models.wrappers.HuggingFaceModelWrapper(
                model, model.tokenizer
            )
            model.to('cuda')
        else:
            model = textattack.models.wrappers.PyTorchModelWrapper(
                model, model.tokenizer
            )
            model.to('cuda')
    else:
        if (model_name in HUGGINGFACE_MODELS):            
            # Support loading models automatically from the HuggingFace model hub.
            model_name = HUGGINGFACE_MODELS[model_name]
        
        colored_model_name = textattack.shared.utils.color_text(
            model_name, color="blue", method="ansi"
        )
        textattack.shared.logger.info(
            f"Loading pre-trained model from HuggingFace model repository: {colored_model_name}"
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
#        model.cuda(device = 0)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        model.to('cuda')
    return model    
