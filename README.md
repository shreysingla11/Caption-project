# Extending Generative Latent Optimization for Image Captioning

![image](https://user-images.githubusercontent.com/22210756/167075363-135ce228-1282-402b-b041-11a328a75b79.png)

TL:DR; -- Using generative models to obtain the image embeddings should be better than standard end-to-end encoder-decoder models for image captioning. The idea is that generative image models are better at grouping similar images together. This may or may not happen with standard encoder-decoder models

## Results 

On Flickr8K image captioning, best results obtained using GPT-J as the text decoder 

| Model              | BLEU |
|--------------------|------|
| CNN + RNN          | 7.9  |
| CNN + LSTM         | 11.9 |
| CNN + Transformers | 50.8 |
| GLO + RNN          | 15.9 |
| GLO + LSTM         | 22.1 |
| GLO + Transformers | 66.7 |

The proposed framework can also be unintentionally used for text-to-image synthesis as follows :

1. Sample a latent vector
2. Optimize this latent vector till the text decoder outputs the required caption (threshold on BLEU)
3. Use the optimized latent vector on the image decoder to obtain results 

For caption, "A girl in blue dress sitting on the ground eating cotton candy" following images were generated (GLO done with 5 random starting vectors)

![image](https://user-images.githubusercontent.com/22210756/167076063-ad42d7e9-711a-440b-883d-424ed298f204.png)

Despite never being trained for text-to-image synthesis task, our model generates images that are close to the given subject "A girl in blue dress sitting". All the generated models missed the "cotton candy" part but perhaps more insight will be offered by scaling this experiment up. Very intriguing to see generated images being at least somewhat related to the caption without any explicit training !

## Contributors

* Bhavnoor Singh Marok (190050027)
* Kanad Pardeshi (190050056)
* Shrey Singla (190050114)
* Ashutosh Sathe (21q050012)
