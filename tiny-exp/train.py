import torch 
import torch.nn as nn
from torchvision import transforms
import numpy as np 
from transformers import GPTJConfig, GPTJModel, Trainer, TrainingArguments, AutoTokenizer, set_seed
from model import GLOCaptions
from glo import Generator, Flickr8K

DATA_ROOT = '../../../data/'

GLO_CKPT = '../../g_epoch=1550.pt'
GLO_DIM = 512

DECODER_CONFIG = GPTJConfig(
    n_positions=128,
    n_embd=512,
    n_layer=12,
    n_head=16
) # ~63.5M parameters
DECODER_PRETRAINED = None # Can be replaced with `EleutherAI/gpt-j-6B`

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

num_train = 250
save_steps = 1000
save_total_limit = 3
evaluation_strategy = 'epoch'
learning_rate = 1e-5 
batch_size = 16

training_args = TrainingArguments(
    output_dir=f'./xformer_ckpts/',
    overwrite_output_dir=True, #YOLO
    num_train_epochs=num_train,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size, # Can theoretically do 2xbatch_size since gradients are disabled, but eh
    report_to=['tensorboard'] # I hate you wandb for not syncing
    save_total_limit=save_total_limit,
    evaluation_strategy=evaluation_strategy,
    learning_rate=learning_rate,
    weight_decay=1e-5,
    warmup_steps=500,
)
def collate_fn(feat):
    ret = {}
    images = torch.from_numpy(np.stack([f[0] for f in feat]))
    captions = [f[1] for f in feat]
    token_output = tokenizer(captions, padding=True, return_tensors='pt')
    ret['images'] = images 
    ret['attention_mask'] = token_output['attention_mask']
    ret['decoder_input_ids'] = token_output['input_ids']
    return ret

def main():
    glo = Generator(GLO_DIM)
    glo.load_state_dict(torch.load(GLO_CKPT))
    glo.eval()

    decoder = GPTJModel(DECODER_CONFIG) if DECODER_PRETRAINED is None else GPTJModel.from_pretrained(DECODER_PRETRAINED)

    transforms = transform=transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    model = GLOCaptions(glo, decoder)

    train = Flickr8K(root_dir=DATA_ROOT, mode='train', transform=transforms)
    dev = Flickr8K(root_dir=DATA_ROOT, mode='dev', transform=transforms)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=collate_fn,
    )
    result = trainer.train()
    trainer.save_model()
    trainer.log_metrics('train', result.metrics)
    trainer.save_metrics('train', result.metrics)
    trainer.save_state()

if __name__ == '__main__':
    main()
