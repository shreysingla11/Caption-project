# GLO-Captions

## Training and evaluation

1. Train the GLO model first `python glo.py ./ -o vis/d512 -gpu -d 512 -e 1600`
2. Train the decoder using `python train.py` -- modify the hyperparameters and the checkpoint path
3. Evaluate using `python eval.py`
