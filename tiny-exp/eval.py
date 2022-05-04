import numpy as np 
import torch 
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer 
import datasets
from tqdm import tqdm
from glo import Flickr8K

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
# Alternatively, load individual checkpoints, this should work since `GLOCaptions` is basically a `transformers.PreTrainedModel`
model = AutoModel.from_pretrained('./xformer_ckpts/') 

metric = datasets.load_metric('bleu', keep_in_memory=True)

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test = Flickr8K(root_dir='../../../data/', mode='test', transform=transform) 
with torch.no_grad():
    for i, sample in enumerate(tqdm(test)):
        image, text = sample 
        z = model.embed_func(image, model.embed_args)
        gen_text = tokenizer.batch_decode(model.decoder.generate(inputs_embeds=z, do_sample=True, temperature=0.9, max_length=100))[0]
        # [text] is important for BLEU implementation
        metric.add(prediction=gen_text, reference=[text]) 

print(f'BLEU: {metric.compute()}')
        
