import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import time as t
import seaborn as sns

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToTensor, Normalize, Compose


np.random.seed(0)
torch.manual_seed(0)


MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
MONTHS_WITHOUT_WINTER = ['03', '04', '05', '06', '07', '08', '09', '10']
LOCATIONS = {
    'Alpnach': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
    # 'Bern1': ['01', '02', '03', '04', '05', '06'],
    # 'Bern2': ['01', '02'],
    # 'NE': ['05', '06', '07', '08', '09']
}
YEAR = '2018'
EXPO = '10'

# ------------------------------------------------ Loading the data ----------------------------------------------------


def read_data():
    """
    Reads image and CSV data for a list of months and a specific location.

    Returns:
        tuple: A tuple with images as a NumPy array and combined dataset columns.
    """

    # Initialize lists for images and data
    all_images = []
    all_data = []

    for location, months in LOCATIONS.items():
        for month in months:
            # Paths for the dataset and images
            csv_path = f'../datasets/tsi_dataset/expo{EXPO}_{location}{YEAR}/{month}_{YEAR}_complete_exposure{EXPO}/{month}_{YEAR}_expo{EXPO}_resized.csv'
            dir_path = f'../datasets/tsi_dataset/expo{EXPO}_{location}{YEAR}/{month}_{YEAR}_complete_exposure{EXPO}/resized/'

            # Read the dataset CSV
            try:
                dataset = pd.read_csv(csv_path)
            except FileNotFoundError:
                print(f"Error: CSV file not found at {csv_path}")
                continue

            # Initialize list to store images
            image_list = []

            # Load each image in the directory
            try:
                for filename in os.listdir(dir_path):
                    route = os.path.join(dir_path, filename)
                    img = cv2.imread(route, 1)  # matrix (NumPy array) containing pixel intensity values
                    img = cv2.resize(img, (224, 224))  # resize images
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # alter channels from BGR to RGB
                    # print(img.shape)
                    if img is not None:
                        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        image_list.append(img)
                    else:
                        print(f"Warning: Unable to read image {filename}")

                # Convert list to NumPy array
                image_list = np.array(image_list)
            except FileNotFoundError:
                print(f"Error: Image directory not found at {dir_path}")
                continue

            all_images.append(image_list)
            all_data.append(dataset[['Irradiance', 'Zenith', 'Temperature', 'Humidity', 'Pressure', 'Hour']])

    if all_images:
        all_images = np.concatenate(all_images, axis=0)
    else:
        all_images = None

    if all_data:
        all_data = pd.concat(all_data, ignore_index=True)
    else:
        all_data = None

    return all_images, all_data


images, df = read_data()

dataframe = np.array(df)
del df

# ---------------------------------------------- Sequences creation ----------------------------------------------------

sequence_length = 5
step = 1
sequences_df = np.array([dataframe[i:i + sequence_length] for i in range(0, len(dataframe) - sequence_length + 1, step)])
sequences_im = np.array([images[i:i + sequence_length] for i in range(0, len(images) - sequence_length + 1, step)])
first_elements = np.array(sequences_df[:, sequence_length - 1, 0])  # vytiahnute Irradiance z posledneho obrazku kazdej sekvencie
del dataframe
del images

# ------------------------------------------ Features & Labels alignment -----------------------------------------------

first_elements = first_elements[1:]  # odstranena prva hodnota Irradiance

sequences_im = sequences_im[:-1]  # odstraneny posledny obrazok
sequences_df = sequences_df[:-1]

# ------------------------------------ Removing outliers & cross-day sequences -----------------------------------------

# Create a mask to filter out unwanted indices
mask = [
    not (abs(seq[0][-1] - seq[sequence_length - 1][-1]) > 3 or first_elements[idx] > 950)
    for idx, seq in enumerate(sequences_df)
]

# Apply the mask to filter data structures
y = first_elements[mask]
X_im = sequences_im[mask]
sequences_df = sequences_df[mask]
del mask

X_df = sequences_df[:, :, :-1]  # hour values removed , irradiance + other features values leave as 3D array

# ------------------------------------------------ Sets creation -------------------------------------------------------

del sequences_im
del sequences_df
del first_elements

# num_samples = sequences_im.shape[0]
# indices = np.arange(num_samples)
#
# train_idx, temp_idx = train_test_split(
#     indices,
#     train_size=0.8,
#     shuffle=True,
#     random_state=10
# )
#
# valid_idx, test_idx = train_test_split(
#     temp_idx,
#     train_size=0.5,
#     shuffle=True,
#     random_state=10
# )
#
# X_train_im = sequences_im[train_idx]
# X_valid_im = sequences_im[valid_idx]
# X_test_im = sequences_im[test_idx]
#
# X_train_df = sequences_df[train_idx]
# X_valid_df = sequences_df[valid_idx]
# X_test_df = sequences_df[test_idx]
#
# y_train = first_elements[train_idx]
# y_valid = first_elements[valid_idx]
# y_test = first_elements[test_idx]
# del sequences_im
# del sequences_df
# del first_elements
# del train_idx
# del temp_idx
# del valid_idx
# del test_idx
# del indices
#
# print(50*"*" + " ViT data " + 50*"*")
# print(f'Train set: {len(X_train_im)} samples, Test set: {len(X_test_im)} samples, Valid set: {len(X_valid_im)} samples')
# print(50*"*" + "  shapes  " + 50*"*")
# print(f'{X_train_im.shape} {X_test_im.shape} {X_valid_im.shape}')
#
# print(47*"*" + " Numerical data " + 47*"*")
# print(f'Train set: {len(X_train_df)} samples, Test set: {len(X_test_df)} samples, Valid set: {len(X_valid_df)} samples')
# print(47*"*" + "     shapes     " + 47*"*")
# print(f'{X_train_df.shape} {X_test_df.shape} {X_valid_df.shape}')

# -------------------------------------------- Image data processing ---------------------------------------------------


# Initialize accumulators
mean_im = np.zeros(3, dtype=np.float64)
std_im = np.zeros(3, dtype=np.float64)

# Iterate over batches to avoid memory overload
for i in range(3):  # For each channel
    channel_data = X_im[:, :, :, :, i]  # Extract single channel
    mean_im[i] = np.mean(channel_data)  # Compute mean
    std_im[i] = np.std(channel_data)  # Compute std deviation


# Assuming mean and std are computed on raw images (0-255 scale)
mean_im = mean_im / 255.0
std_im = std_im / 255.0
print("Computed Mean:", mean_im)
print("Computed Std:", std_im)

# Transformation with normalization
transform = Compose([
    ToTensor(),
    Normalize(mean=mean_im, std=std_im)
])
del mean_im
del std_im


X_im = [
    torch.stack([transform(image) for image in batch]) for batch in X_im
]
# X_valid_im = [
#     torch.stack([transform(image) for image in batch]) for batch in X_valid_im
# ]
# X_test_im = [
#     torch.stack([transform(image) for image in batch]) for batch in X_test_im
# ]
del transform

# ------------------------------------------ Numerical data processing -------------------------------------------------


# before: (12004, 5, 1) , but MinMaxScaler expects dim <= 2 , found 3
scaler = MinMaxScaler()
X_df = scaler.fit_transform(X_df.reshape(len(X_df), -1))
# X_valid_df = scaler.transform(X_valid_df.reshape(len(X_valid_df), -1))
# X_test_df = scaler.transform(X_test_df.reshape(len(X_test_df), -1))
# (12004, 5)
X_df = torch.tensor(X_df, dtype=torch.float32)
# X_valid_df = torch.tensor(X_valid_df, dtype=torch.float32)
# X_test_df = torch.tensor(X_test_df, dtype=torch.float32)
# torch.Size([12004, 5])

# ----------------------------------------------------------------------------------------------------------------------
#                                                OWN DATASET PART
# ----------------------------------------------------------------------------------------------------------------------


class CombinedDataset(Dataset):
    def __init__(self, image_list, number_list, label_list):
        self.image_list = image_list
        self.number_list = number_list
        self.label_list = torch.tensor(label_list, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.image_list[idx], self.number_list[idx], self.label_list[idx]

# ----------------------------------------------------------------------------------------------------------------------
#                                                   MODEL PART
# ----------------------------------------------------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=64, device='cuda'):
        super(MLP, self).__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.mlp = nn.Sequential(
            # nn.Flatten(),  # Flatten: (N, 5, 1) -> (N, 5)  nepotrebne kvoli tomu ze uz po aplikacii scaleru minmax su v shape (N, 5)
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),  # There is no ReLU, because of the regression problem (we want both negative or positive numbers)
            # nn.ReLU()
        ).to(self.device)

    def forward(self, symptoms):
        return self.mlp(symptoms)


# ----------------------------------------------------------------------------------------------------------------------


def patchify(image_list, n_patches):
    """
    Assume n_patches is 7, image size of (5, 3, 224, 224) and N samples

    Each sub-image of size (3, 224, 224) is divided into patches of size (3, 32, 32)
    After that each patch is flattened into a vector of size 3 * 32 * 32 = 3072 , 3072-dimensional vector
    For each image, we got 5 * 7 * 7 = 245 patches of size 3072

    :param image_list: batch of images acquired from the dataloader
    :param n_patches: number of patches per side
    :return: patches, resulted shape is (N, 245, 3072)
    """
    n, seq_len, c, h, w = image_list.shape

    assert h == w, "Patchify method is implemented for square images only"

    patch_size = h // n_patches

    image_list = image_list.view(n * seq_len, c, h, w)
    patches = image_list.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # patch extraction
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(n * seq_len, n_patches ** 2, -1)  # flatten
    patches = patches.view(n, seq_len * n_patches ** 2, -1)  # (N, seq_len * n_patches², patch_size² * C)

    return patches

# ----------------------------------------------------------------------------------------------------------------------


def get_positional_encoding(seq_len, hidden_d):
    """
    Generate positional encoding for a given sequence length and hidden dimensionality.

    :param seq_len: (int) Length of the sequence.
    :param hidden_d: (int) Dimensionality of the token embeddings.
    :return: torch.Tensor: Positional encoding of shape (seq_len, hidden_d).
    """
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # Containing values representing positions, then reshaped from (seq_len,) to (seq_len, 1)
    div_term = torch.exp(
        torch.arange(0, hidden_d, 2, dtype=torch.float) * (-np.log(10000.0) / hidden_d))  # (hidden_d/2,)

    # Compute sinusoidal positional encoding
    positional_encoding = torch.zeros((seq_len, hidden_d), dtype=torch.float)
    positional_encoding[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
    positional_encoding[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

    return positional_encoding

# ----------------------------------------------------------------------------------------------------------------------


class MyMSA(nn.Module):
    def __init__(self, hidden_d, n_heads=2, device='cuda'):
        """
        Multi-Head Self-Attention (MHSA) implementation for regression.

        :param hidden_d: (int) Dimensionality of token embeddings.
        :param n_heads: (int) Number of attention heads.

        - Batch Processing: The original code processes each sequence and head independently using nested loops, which
        is inefficient. The optimized code uses vectorized operations for all sequences and heads simultaneously,
        leveraging PyTorch's GPU acceleration.
        - Use of Single Linear Layers: Instead of creating separate Linear layers for each head, a single Linear layer
        is used, and the outputs are reshaped to split into multiple heads. This reduces the number of parameters and
        simplifies the code.
        - Reshaping for Heads: The reshaping (view) and transposing (transpose) steps manage the division of dimensions
        into multiple heads without manual slicing.
        - Concatenation Simplified: The concatenation of results from different heads is done in a single step using .
        view and .transpose, eliminating the need for manual looping and stacking.

        Expected benefits :
        - Performance: The vectorized approach minimizes Python overhead and leverages optimized matrix operations,
        leading to faster computation.
        - Readability: The code is cleaner, with reduced complexity and clearer operations.
        - Scalability: The new implementation scales better with larger batch sizes and sequence lengths.

        """
        super(MyMSA, self).__init__()
        self.hidden_d = hidden_d  # total dimensionality of the input tokens
        self.n_heads = n_heads  # the required number of attention heads
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        assert self.hidden_d % self.n_heads == 0, f"Can't divide dimension {self.hidden_d} into {self.n_heads} heads"

        self.d_head = self.hidden_d // self.n_heads  # Dimensionality per head

        # Linear mappings for query, key, and value for each head
        # These operate on the entire input dimensionality hidden_d for simplicity
        self.q_mappings = nn.Linear(self.hidden_d, self.hidden_d).to(self.device)
        self.k_mappings = nn.Linear(self.hidden_d, self.hidden_d).to(self.device)
        self.v_mappings = nn.Linear(self.hidden_d, self.hidden_d).to(self.device)

        self.softmax = nn.Softmax(dim=-1)  # compute the attention scores as probabilities

    def forward(self, sequences):
        """
        Forward pass for MHSA.

        :param sequences: (torch.Tensor) Input tensor of shape (N, seq_length, d).
        :return: (torch.Tensor) Output tensor of shape (N, seq_length, d).
        """
        n_samples, seq_length, _ = sequences.shape

        # Linear projections for query, key, value
        # Each token is mapped to query, key and value vectors using linear transformations - out has the same shape as the input (n_samples, seq_len, dim)
        # view splits the last dimension dim into n_heads of lengths d_head, shape should be now (n_samples, seq_len, n_heads, d_head)
        # transpose swaps seq_len with n_heads , shape should be now (n_samples, n_heads, seq_len, d_head)
        q = self.q_mappings(sequences).view(n_samples, seq_length, self.n_heads, self.d_head).transpose(1, 2)  # (n_samples, n_heads, seq_length, d_head)
        k = self.k_mappings(sequences).view(n_samples, seq_length, self.n_heads, self.d_head).transpose(1, 2)  # (n_samples, n_heads, seq_length, d_head)
        v = self.v_mappings(sequences).view(n_samples, seq_length, self.n_heads, self.d_head).transpose(1, 2)  # (n_samples, n_heads, seq_length, d_head)

        # Scaled dot-product attention
        # (self.d_head ** 0.5) stabilizes gradients and avoids exploding attention values for larger d_head
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)  # (n_samples, n_heads, seq_length, seq_length)
        attention_probs = self.softmax(attention_scores)  # (n_samples, n_heads, seq_length, seq_length)

        # Attention-weighted values
        attention_output = torch.matmul(attention_probs, v)  # (n_samples, n_heads, seq_length, d_head)

        # Concatenate heads and project output
        attention_output = attention_output.transpose(1, 2).contiguous()  # (n_samples, seq_length, n_heads, d_head)
        attention_output = attention_output.view(n_samples, seq_length, self.hidden_d)  # (n_samples, seq_length, d)

        return attention_output

# ----------------------------------------------------------------------------------------------------------------------


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4, device='cuda'):
        """
        Block for Vision Transformer Encoder.

        :param hidden_d: (int) Dimensionality of each token.
        :param n_heads: (int) Number of attention heads.
        :param mlp_ratio: (int) Multiplier applied to hidden dimensionality for MLP.
        :return:
        """
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Layer Normalization
        self.norm1 = nn.LayerNorm(self.hidden_d).to(self.device)

        # Multi-Head Self-Attention
        self.mhsa = MyMSA(self.hidden_d, self.n_heads).to(self.device)

        # Layer Normalization
        self.norm2 = nn.LayerNorm(self.hidden_d).to(self.device)

        # Fully Connected Layer (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, mlp_ratio * self.hidden_d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_ratio * self.hidden_d, self.hidden_d),
            nn.Dropout(0.1)
        ).to(self.device)

    def forward(self, x):
        """
        Forward pass through the encoder block.

        :param x: (torch.Tensor) Input tensor of shape (N, S, D)
          where N is batch size, S is sequence length, D is token dimensionality.
        :return: (torch.Tensor) Normalized tokens of shape (N, S, D)
        """
        # Apply Layer Normalization
        normalized_tokens = self.norm1(x)

        # Apply Multi-Head Self-Attention
        # - each patch get updated based on some similarity measure with the other patches
        # - we first linearly map each patch to 3 distinct vectors: query, key, value
        # - we then compute 'attention score' between patch's query and all other patches' keys, divide by the sqrt of the dimensionality of these vectors (our example sqrt(1028) )
        # - we then apply a softmax to get a probability distribution over all patches
        # - we then multiply each softmaxed attention score with the value vectors associated with the different k vectors and sum all up
        # - finally each patch assumes a new value that is based on its similarity with other patches
        mhsa_output = self.mhsa(normalized_tokens.to(self.device))

        # Apply Residual Connections
        res_con_out = x + mhsa_output

        # Apply Layer Normalization
        normalized_tokens_two = self.norm2(res_con_out)

        # Apply MLP
        res_con_out_two = res_con_out + self.mlp(normalized_tokens_two)

        return res_con_out_two

# ----------------------------------------------------------------------------------------------------------------------


class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=6, hidden_d=256, n_heads=2, device='cuda'):
        """
        Vision Transformer.

        :param chw: Image dimensions (channels, height, width)
        :param n_patches: (int) Number of patches in the image.
        :param n_blocks: (int) Number of ViT blocks.
        :param hidden_d: (int) Dimensionality of each token.
        :param n_heads: (int) Number of attention heads.
        :return:
        """
        # Super constructor
        super(MyViT, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Input and patches sizes
        assert (
                chw[1] % self.n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
                chw[2] % self.n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] // self.n_patches, chw[2] // self.n_patches)  # (32, 32)

        # 1) Linear mapper
        self.patch_dim = int(chw[0] * self.patch_size[0] * self.patch_size[1])  # 3072
        self.linear_mapper = nn.Linear(self.patch_dim, self.hidden_d).to(
            self.device)  # we run (N, 245, 3072) tensor through a (3072, 8) linear mapper (or matrix)

        # 2) Learnable class token
        # first dimension `1` is for the batch-like setting, making it easy to expand this token for any batch
        # second dimension `1` signifies that this is a single, unique token (not multiple tokens)
        # third dimension `hidden_d` represents the embedding dimension for the token, matching other hidden representations in the model
        self.class_token = nn.Parameter(torch.randn(1, 1, self.hidden_d,
                                                    device=self.device))  # nn.Parameter wraps the tensor do it's treated as a model parameter, meaning it will be optimized during training

        # 3) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [MyViTBlock(self.hidden_d, self.n_heads).to(self.device) for _ in range(self.n_blocks)]
        )

        # # 4) Regression MLP (Classification MLPk)
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.hidden_d, out_d)  # sem dam hidden_d  dim vystupnej vrstvy num. modela
        # ).to(self.device)  # nn.Sequential(nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1))

    def forward(self, image_list):
        n, seq_len, c, h, w = image_list.shape  # n == batch_size == 128

        patches = patchify(image_list, self.n_patches).to(self.device)  # .to(self.positional_embeddings.device)
        # print(f'patches shape : {patches.shape}')  # torch.Size([N, 245, 3072])

        # Now that we have our flattened patches, we can map each of them through a Linear mapping.
        # Thus, we add a parameter called hidden_d for 'hidden dimension'.
        # We will thus be mapping each 3072 dimensional patch to a hidden_d dimensional patch.

        patch_tokens = self.linear_mapper(patches)
        # print(f'tokens shape : {patch_tokens.shape}')  # torch.Size([N, 245, 1028])

        # A special token that we add to our model has the role of capturing information about the other tokens.
        # When information about all other tokens will be present here, we will be able to capture global context across each patches in an image or sequence, effectively condensing input details for final prediction.
        # The initial value of the special token is a parameter of the model that needs to be learned.
        # We will add a parameter to our model and convert our (N, 245, 8) tokens tensor to an (N, 246, 8) tensor.

        class_token = self.class_token.expand(n, 1, -1)  # n images will have the same class token
        tokens = torch.cat((class_token, patch_tokens), dim=1).to(
            self.device)  # add special token to the beginning allowing the model to learn a representation of the entire sequence
        # print(f'tokens shape after class token added : {tokens.shape}')  # torch.Size([N, 246, 1028])

        # Positional encoding allows the model to understand where each patch would be placed in the original image.
        # Previous work by Vaswani et. al. suggests that we can learn such positional embeddings by adding sines and cosines waves.
        # In particular, positional encoding adds high-frequency values to the first dimensions and low-frequency values to the latter dimensions.
        # in each sequence, for token i we add to its j-th coordinate the following value:
        # p_i,j = { sin(i / 10000^(j / d_emb-dim)), if j is even; cos(i / 10000^(j-1 / d_emb-dim)), if j is odd }
        # This positional embeddings is a function of the number of elements in the sequence and the dimensionality of each element.
        # Thus, it is always a 2-dimensional tensor or "rectangle".

        pos_encoding = get_positional_encoding(tokens.size(1), self.hidden_d).to(self.device)
        tokens += pos_encoding
        # Second way using precomputed and registered embeddings
        # out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        # print(f'tokens shape after positional encoding added : {tokens.shape}')  # torch.Size([N, 246, 1028])

        # Transformer Blocks
        for block in self.blocks:
            tokens = block(tokens)
        # print(f'tokens shape after transformer blocks : {tokens.shape}')  # torch.Size([N, 246, 1028])

        # Getting the classification token only
        token = tokens[:, 0]

        # return self.mlp(token)
        return token

# ----------------------------------------------------------------------------------------------------------------------


class CombinedModel(nn.Module):
    def __init__(self, vit_model, mlp_model, vit_dim, mlp_dim, out_d=1, device='cuda'):
        super(CombinedModel, self).__init__()
        self.vit_model = vit_model
        self.mlp_model = mlp_model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 4) Regression MLP (Classification MLPk)
        self.mlp = nn.Sequential(
            nn.LayerNorm(vit_dim + mlp_dim),
            nn.ReLU(),
            nn.Linear(vit_dim + mlp_dim, out_d)
        ).to(self.device)

    def forward(self, image_list, number_list):
        vit_token = self.vit_model(image_list)
        mlp_vector = self.mlp_model(number_list)

        return self.mlp(torch.cat((vit_token, mlp_vector), dim=1))  # torch.Size([32, 512 + 64])

# ----------------------------------------------------------------------------------------------------------------------
#                                                   MAIN PART
# ----------------------------------------------------------------------------------------------------------------------


def compute_metrics(outputs, label_list, epsilon=1e-8):
    """Computes RMSE, MAE, %MAE and R² metrics."""
    errors = outputs - label_list
    mse = torch.mean(errors ** 2)
    mae = torch.mean(torch.abs(errors))
    rmse = torch.sqrt(mse)
    mape = torch.mean(torch.abs(errors / (label_list + epsilon))) * 100

    total_variance = torch.mean((label_list - torch.mean(label_list)) ** 2)
    explained_variance = total_variance - mse
    r2 = explained_variance / total_variance

    return rmse.item(), mae.item(), mape.item(), r2.item()


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    train_outputs, train_labels = [], []
    for image_list, num_list, label_list in train_loader:
        image_list, num_list, label_list = image_list.to(device), num_list.to(device), label_list.to(device)

        optimizer.zero_grad()  # zero the gradients

        outputs = model(image_list, num_list)  # forward pass
        loss = criterion(outputs, label_list)  # computing loss

        loss.backward()  # backward pass
        optimizer.step()  # update weights

        running_loss += loss.item()
        train_outputs.append(outputs.detach())
        train_labels.append(label_list)

    avg_train_loss = running_loss / len(train_loader)

    # Concatenate train outputs and targets for the epoch
    train_outputs = torch.cat(train_outputs, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    rmse, mae, mape, r2 = compute_metrics(train_outputs, train_labels)

    return avg_train_loss, rmse, mae, mape, r2


# both validate and test
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    valid_outputs, valid_labels = [], []
    with torch.no_grad():
        for image_list, num_list, label_list in loader:
            image_list, num_list, label_list = image_list.to(device), num_list.to(device), label_list.to(device)

            outputs = model(image_list, num_list)  # forward pass
            loss = criterion(outputs, label_list)  # computing loss

            running_loss += loss.item()
            valid_outputs.append(outputs)
            valid_labels.append(label_list)

    avg_valid_loss = running_loss / len(loader)

    # Concatenate valid outputs and targets for the epoch
    valid_outputs = torch.cat(valid_outputs, dim=0)
    valid_labels = torch.cat(valid_labels, dim=0)

    rmse, mae, mape, r2 = compute_metrics(valid_outputs, valid_labels)

    return avg_valid_loss, rmse, mae, mape, r2


# def main():
#     # Hyperparameters
#     n_epochs = 200
#     learning_rate = 0.0005
#     weight_decay = 0.0001
#     patience = 10  # number of epochs to wait before early stopping if no improvement
#     n_patches = 7
#     n_blocks = 8
#     hidden_d = 256
#     n_heads = 2
#     mlp_d = 64
#     batch_sizes = [16, 32, 64, 128]  # Different batch sizes to test
#
#     train_dataset = CombinedDataset(X_train_im, X_train_df, y_train)
#     valid_dataset = CombinedDataset(X_valid_im, X_valid_df, y_valid)
#
#     # Initialize dictionary to store RMSE values for each batch size
#     results = {batch_size: [] for batch_size in batch_sizes}
#
#     for batch_size in batch_sizes:
#         print(f"\nTraining with batch size {batch_size}")
#
#         # Initialize model, optimizer, and loss function
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#         # Defining model, loss function and optimizer
#         print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
#
#         vit_model = MyViT((3, 224, 224), n_patches=n_patches, n_blocks=n_blocks, hidden_d=hidden_d, n_heads=n_heads).to(device)
#         mlp_model = MLP(input_dim=X_train_df.shape[1], output_dim=mlp_d).to(device)
#         model = CombinedModel(vit_model, mlp_model, vit_dim=hidden_d, mlp_dim=mlp_d, out_d=1).to(device)
#
#         optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#         criterion = MSELoss()
#
#         # Dataloader setup
#         train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
#         val_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
#
#         # Early stopping setup
#         best_valid_loss = np.inf
#         patience_counter = 0
#
#         for epoch in range(n_epochs):
#             # Training step
#             train_loss, train_rmse, train_mae, train_mape, train_r2 = train(model, train_loader, criterion, optimizer,
#                                                                             device)
#
#             # Validation step
#             valid_loss, valid_rmse, valid_mae, valid_mape, valid_r2 = evaluate(model, val_loader, criterion, device)
#             results[batch_size].append(valid_rmse)  # Store RMSE for analysis
#
#             print(f"Epoch {epoch + 1}/{n_epochs} - Batch {batch_size}: RMSE {valid_rmse:.4f}")
#
#             # Early Stopping
#             if valid_loss < best_valid_loss:
#                 best_valid_loss = valid_loss
#                 patience_counter = 0  # Reset counter if improvement
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     print(f"Early stopping triggered at epoch {epoch + 1} for batch size {batch_size}.")
#                     break  # Stop training for this batch size
#
#     # Convert results to DataFrame
#     data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
#
#     plot_filepath = f'../eda/box_plot_{t.localtime().tm_year}-{t.localtime().tm_mon}-{t.localtime().tm_mday}_{t.localtime().tm_hour}-{t.localtime().tm_min}-{t.localtime().tm_sec}.png'
#
#     # Plot box plot
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(data=data)
#     plt.xlabel("Batch Size")
#     plt.ylabel("Validation RMSE")
#     plt.title("Impact of Batch Size on Model Performance")
#     plt.savefig(plot_filepath, dpi=300)
#     plt.show()
#
#
# main()


def main():
    # Hyperparameters
    n_epochs = 500
    learning_rate = 0.0005
    weight_decay = 0.0001
    patience = 10  # number of epochs to wait before early stopping if no improvement
    n_patches = 7
    n_blocks = 8
    hidden_d = 256
    n_heads = 2
    mlp_d = 64
    batch_sizes = [16, 32, 64, 128]  # Different batch sizes to test
    # batch_size = 16
    n_splits = 10  # Number of folds

    kf = KFold(n_splits=n_splits, shuffle=False)
    # kf = TimeSeriesSplit(n_splits=n_splits)
    results = {bs: [] for bs in batch_sizes}

    dataset = CombinedDataset(X_im, X_df, y)
    print(f"Dataset length:{dataset.__len__()}")
    for batch_size in batch_sizes:
        print(f"\nTraining with batch size {batch_size}...")

        fold_rmse_scores = []

        for fold, (train_idx, valid_idx) in enumerate(kf.split(dataset)):
            print(f"Fold {fold}:")
            print(f"  Training dataset index: {train_idx}")
            print(f"  Valid dataset index: {valid_idx}")

            train_dataset = Subset(dataset, train_idx)
            valid_dataset = Subset(dataset, valid_idx)

            # Device setup
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Defining model, loss function and optimizer
            print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

            vit_model = MyViT((3, 224, 224), n_patches=n_patches, n_blocks=n_blocks, hidden_d=hidden_d, n_heads=n_heads).to(device)
            mlp_model = MLP(input_dim=X_df.shape[1], output_dim=mlp_d).to(device)
            model = CombinedModel(vit_model, mlp_model, vit_dim=hidden_d, mlp_dim=mlp_d, out_d=1).to(device)

            optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = MSELoss()

            # Dataloader setup
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
            val_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
            # print(f"Fold {fold}: Train size = {len(train_loader.dataset)}, Val size = {len(val_loader.dataset)}")

            # Early stopping setup
            best_valid_loss = np.inf
            patience_counter = 0

            # Lists to plot metrics
            # train_rmse_list, valid_rmse_list = [], []
            # train_loss_list, valid_loss_list = [], []
            # train_mae_list, valid_mae_list = [], []
            # train_mape_list, valid_mape_list = [], []
            # train_r2_list, valid_r2_list = [], []

            for epoch in range(n_epochs):
                train_loss, train_rmse, train_mae, train_mape, train_r2 = train(model, train_loader, criterion, optimizer, device)
                # train_loss_list.append(train_loss)
                # train_rmse_list.append(train_rmse)
                # train_mae_list.append(train_mae)
                # train_mape_list.append(train_mape)
                # train_r2_list.append(train_r2)
                # print(f"Epoch [{epoch+1}/{n_epochs}] Train Loss: {train_loss:.4f} RMSE: {train_rmse:.4f} MAE: {train_mae:.4f} %MAE: {train_mape:.4f} R²: {train_r2:.4f}")

                valid_loss, valid_rmse, valid_mae, valid_mape, valid_r2 = evaluate(model, val_loader, criterion, device)
                # valid_loss_list.append(valid_loss)
                # valid_rmse_list.append(valid_rmse)
                # valid_mae_list.append(valid_mae)
                # valid_mape_list.append(valid_mape)
                # valid_r2_list.append(valid_r2)
                # print(f"Epoch [{epoch+1}/{n_epochs}] Validation Loss: {valid_loss:.4f} RMSE: {valid_rmse:.4f} MAE: {valid_mae:.4f} %MAE: {valid_mape:.4f} R²: {valid_r2:.4f}")

                print(f"  Epoch {epoch+1}: Train RMSE {train_rmse:.4f} | Valid RMSE {valid_rmse:.4f}")
                # fold_rmse_scores.append(valid_rmse)

                # Early stopping check
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0  # reset counter if validation loss is improved
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch + 1} for batch size {batch_size}.")
                        fold_rmse_scores.append(valid_rmse)
                        break
                if epoch == n_epochs - 1:
                    fold_rmse_scores.append(valid_rmse)

            results[batch_size].extend(fold_rmse_scores)

    # Convert results to DataFrame
    data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))

    plot_filepath = f'../eda/box_plot_{t.localtime().tm_year}-{t.localtime().tm_mon}-{t.localtime().tm_mday}_{t.localtime().tm_hour}-{t.localtime().tm_min}-{t.localtime().tm_sec}.png'

    # Plot box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.xlabel("Batch Size")
    plt.ylabel("Validation RMSE")
    plt.title("Impact of Batch Size on Model Performance")
    plt.savefig(plot_filepath, dpi=300)
    plt.show()


main()
