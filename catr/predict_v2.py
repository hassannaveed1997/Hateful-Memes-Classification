import torch
from transformers import BertTokenizer
from PIL import Image
from catr.datasets import coco
from catr.configuration import Config


def create_caption_and_mask(start_token, max_length):
    
    # Initialize caption
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    caption_template[:, 0] = start_token
    
    # Initialize mask
    mask_template = torch.ones((1, max_length), dtype=torch.bool)
    mask_template[:, 0] = False

    return caption_template, mask_template


@torch.no_grad()
def evaluate(model, image, config, caption, cap_mask):
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption


def predict(
    image_path, 
    model=None, 
    tokenizer=None, 
    start_token=None, 
    config=None
    ):
    """
    Predict the caption for a single image
    
    Inputs:
        image_path (str): path to image that caption will be created for
        model (object): torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
        tokenizer (object): BertTokenizer.from_pretrained('bert-base-uncased')
        config (object): catr.configuration.Config()
    """
    
    # Load config
    if config is None:
        config = Config()
    
    # Load captioning model
    if model is None:
        model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)

    # Load tokenizer
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create start token
    if start_token is None:
        start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
        #end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    
    # Preprocess image
    image = Image.open(image_path)
    image = coco.val_transform(image)
    image = image.unsqueeze(0)
    
    # Initialize caption and mask
    caption, cap_mask = create_caption_and_mask(
        start_token, 
        config.max_position_embeddings
        )
    
    # Create caption
    output = evaluate(model, image, config, caption, cap_mask)
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    #result = tokenizer.decode(output[0], skip_special_tokens=True)
    #print(result.capitalize())
    return result.capitalize()