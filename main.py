import plac
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as fnn
from torch.autograd import Variable
from torch.optim import SGD
from torchvision.datasets import LSUN
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid





emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead






def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size,0:size].T)
    gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w), 
    # and since we have depth-separable convolution we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
    n_channels, _, kw, kh = kernel.shape
    img = fnn.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return fnn.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = fnn.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        
    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(
                size=self.k_size, sigma=self.sigma, 
                n_channels=input.shape[1], cuda=input.is_cuda
            )
        pyr_input  = laplacian_pyramid( input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(fnn.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


class IndexedDataset(Dataset):
    """ 
    Wraps another dataset to sample from. Returns the sampled indices during iteration.
    In other words, instead of producing (X, y) it produces (X, y, idx)
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return (img, label, idx)













class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class Generator(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(Generator, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind















def project_l2_ball(z):
    """ project the vectors in z onto the l2 unit norm ball"""
    return z / np.maximum(np.sqrt(np.sum(z**2, axis=1))[:, np.newaxis], 1)


def imsave(filename, array):
    im = Image.fromarray((array * 255).astype(np.uint8))
    im.save(filename)


def main(
        lsun_data_dir: ('Base directory for the LSUN data'),
        image_output_prefix: ('Prefix for image output', 
                              'option', 'o')='glo',
        code_dim: ('Dimensionality of latent representation space', 
                   'option', 'd', int)=2048, 
        epochs: ('Number of epochs to train', 
                 'option', 'e', int)=25,
        use_cuda: ('Use GPU?', 
                   'flag', 'gpu')=False,
        batch_size: ('Batch size', 
                     'option', 'b', int)=128,
        lr_g: ('Learning rate for generator', 
               'option', None, float)=4e-1,
        lr_z: ('Learning rate for representation_space', 
               'option', None, float)=1.,
        max_num_samples: ('Cap on the number of samples from the LSUN dataset', 
                          'option', 'n', int)=-1,
        init: ('Initialization strategy for latent represetation vectors', 
               'option', 'i', str, ['pca', 'random'])='pca',
        n_pca: ('Number of samples to take for PCA',
                'option', None, int)=(64 * 64 * 3 * 2),
        loss: ('Loss type (Laplacian loss as in the paper, or L2 loss)',
               'option', 'l', str, ['lap_l1', 'l2'])='lap_l1',
):

    def maybe_cuda(tensor):
        return tensor.cuda() if use_cuda else tensor

    # train_set = IndexedDataset(
    #     LSUN(lsun_data_dir, classes=['church_outdoor_train'], 
    #          transform=transforms.Compose([
    #              transforms.Resize(64),
    #              transforms.CenterCrop(64),
    #              transforms.ToTensor(),
    #              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #          ]))
    # )
    # train_loader = torch.utils.data.DataLoader(
    #     train_set, batch_size=batch_size, 
    #     shuffle=True, drop_last=True,
    #     num_workers=8, pin_memory=use_cuda,
    # )


    train_set = CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize]))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)



    # val_loader = torch.utils.data.DataLoader(train_set, shuffle=False, batch_size=8*8)
    val_set = CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize]))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)



    if max_num_samples > 0:
        train_set.base.length = max_num_samples
        train_set.base.indices = [max_num_samples]

    # initialize representation space:
    if init == 'pca':
        from sklearn.decomposition import PCA

        # first, take a subset of train set to fit the PCA
        print(n_pca)
        print(train_loader.batch_size)
        print(n_pca // train_loader.batch_size)
        # print(next(train_loader))
        print(iter(train_loader))

        X_pca = np.vstack([
            X.cpu().numpy().reshape(len(X), -1)
            for i, (X, _, _) in zip(range(n_pca // train_loader.batch_size), train_loader)
        ])
        print("perform PCA...")
        pca = PCA(n_components=code_dim)
        pca.fit(X_pca)
        # then, initialize latent vectors to the pca projections of the complete dataset
        Z = np.empty((len(train_loader.dataset), code_dim))
        for X, _, idx in tqdm(train_loader, 'pca projection'):
            Z[idx] = pca.transform(X.cpu().numpy().reshape(len(X), -1))

    elif init == 'random':
        Z = np.random.randn(len(train_set), code_dim)

    Z = project_l2_ball(Z)

    global emb_dim, attention_dim, decoder_dim, dropout

    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)



    g = maybe_cuda(Generator(attention_dim,emb_dim,decoder_dim,len(word_map),code_dim,dropout))
    #loss_fn = LapLoss(max_levels=3) if loss == 'lap_l1' else nn.MSELoss()
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    zi = maybe_cuda(torch.zeros((batch_size, code_dim)))
    zi = Variable(zi, requires_grad=True)
    optimizer = SGD([
        {'params': g.parameters(), 'lr': lr_g}, 
        {'params': zi, 'lr': lr_z}
    ])

    Xi_val, _, idx_val = next(iter(val_loader))
    imsave('target.png',
           make_grid(Xi_val.cpu() / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))

    for epoch in range(epochs):
        losses = []
        progress = tqdm(total=len(train_loader), desc='epoch % 3d' % epoch)

        for i, (Xi, caps, caplens) in enumerate(train_loader):
            Xi = Variable(maybe_cuda(Xi))
            zi.data = maybe_cuda(torch.FloatTensor(Z[idx.numpy()]))

            optimizer.zero_grad()
            rec = g(zi,caps,caplens)
            loss = loss_fn(rec, Xi)
            loss.backward()
            optimizer.step()

            Z[idx.numpy()] = project_l2_ball(zi.data.cpu().numpy())
            print(loss)
            losses.append(loss.item())
            progress.set_postfix({'loss': np.mean(losses[-100:])})
            progress.update()

        progress.close()

        # visualize reconstructions
        rec = g(Variable(maybe_cuda(torch.FloatTensor(Z[idx_val.numpy()]))))
        imsave('%s_rec_epoch_%03d.png' % (image_output_prefix, epoch), 
               make_grid(rec.data.cpu() / 2. + 0.5, nrow=8).numpy().transpose(1, 2, 0))

if __name__ == "__main__":
    plac.call(main)

