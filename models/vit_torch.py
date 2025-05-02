__author__ = "Peter Kopecký"
__email__ = "xkopecky@stuba.sk"

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import time as t

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

np.random.seed(0)
torch.manual_seed(0)

MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
MONTHS_WITHOUT_WINTER = ['03', '04', '05', '06', '07', '08', '09', '10']
LOCATIONS = ['Alpnach']
YEAR = '2018'
EXPO = '10'


# ------------------------------------------------ Loading the data ----------------------------------------------------


def read_data(months, location):
    """
    Reads image and CSV data for a list of months and a specific location.

    Parameters:
        months (list): List of month identifiers (e.g., ["01", "02", "03"]).
        location (str): The location identifier.

    Returns:
        tuple: A tuple with images as a NumPy array and combined dataset columns.
    """
    all_images = []
    all_data = []

    for month in months:
        csv_path = f'../datasets/tsi_dataset/expo{EXPO}_{location}{YEAR}/{month}_{YEAR}_complete_exposure{EXPO}/{month}_{YEAR}_expo{EXPO}_resized.csv'
        dir_path = f'../datasets/tsi_dataset/expo{EXPO}_{location}{YEAR}/{month}_{YEAR}_complete_exposure{EXPO}/resized/'

        try:
            dataset = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            continue

        image_list = []

        try:
            for filename in os.listdir(dir_path):
                route = os.path.join(dir_path, filename)
                img = cv2.imread(route, 1)
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img is not None:
                    image_list.append(img)
                else:
                    print(f"Warning: Unable to read image {filename}")

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


images, df = read_data(MONTHS, LOCATIONS[0])

dataframe = np.array(df)
del df

# ---------------------------------------------- Sequences creation ----------------------------------------------------


sequence_length = 5
step = 1
sequences_df = np.array(
    [dataframe[i:i + sequence_length] for i in range(0, len(dataframe) - sequence_length + 1, step)])
sequences_im = np.array([images[i:i + sequence_length] for i in range(0, len(images) - sequence_length + 1, step)])
first_elements = np.array(sequences_df[:, 4, 0])
del dataframe
del images

# ------------------------------------------ Features & Labels alignment -----------------------------------------------


first_elements = first_elements[1:]

sequences_im = sequences_im[:-1]

sequences_df = sequences_df[:-1]

# ------------------------------------ Removing outliers & cross-day sequences -----------------------------------------


mask = [
    not (abs(seq[0][-1] - seq[sequence_length - 1][-1]) > 3 or first_elements[idx] > 1300)
    for idx, seq in enumerate(sequences_df)
]
del sequences_df

first_elements = first_elements[mask]
sequences_im = sequences_im[mask]
del mask

# ------------------------------------------------ Sets creation -------------------------------------------------------


train_im, temp_im, train_y, temp_y = train_test_split(
    sequences_im,
    first_elements,
    train_size=0.8,
    shuffle=True,
    random_state=10
)
del sequences_im
del first_elements

valid_im, test_im, valid_y, test_y = train_test_split(
    temp_im,
    temp_y,
    train_size=0.5,
    shuffle=True,
    random_state=10
)
del temp_im
del temp_y

print(f'Train set: {len(train_im)} samples, Test set: {len(test_im)} samples, Valid set: {len(valid_im)} samples')
print(47 * "*" + "     shapes     " + 47 * "*")
print(f'{train_im.shape} {valid_im.shape} {test_im.shape}')

# -------------------------------------------- Image data processing ---------------------------------------------------


mean = np.mean(train_im, axis=(0, 1, 2, 3))
std = np.std(train_im, axis=(0, 1, 2, 3))

mean = mean / 255.0
std = std / 255.0
print("Computed Mean:", mean)
print("Computed Std:", std)

transform = Compose([
    ToTensor(),
    Normalize(mean=mean, std=std)
])
del mean
del std

train_im = torch.stack([
    torch.stack([transform(image) for image in batch]) for batch in train_im
])
valid_im = torch.stack([
    torch.stack([transform(image) for image in batch]) for batch in valid_im
])
test_im = torch.stack([
    torch.stack([transform(image) for image in batch]) for batch in test_im
])
del transform


# ----------------------------------------------------------------------------------------------------------------------
#                                              OWN DATASET PART
# ----------------------------------------------------------------------------------------------------------------------


class ImageDataset(Dataset):
    """Custom dataset class for loading images and labels."""

    def __init__(self, image_list, label_list):
        self.images = image_list
        self.labels = torch.tensor(label_list, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ----------------------------------------------------------------------------------------------------------------------
#                                                   MODEL PART
# ----------------------------------------------------------------------------------------------------------------------


def patchify(image_list, n_patches):
    """Divide images into patches and flatten them.

    Assume n_patches is 7, image size of (5, 3, 224, 224) and N samples.

    Each sub-image of size (3, 224, 224) is divided into patches of size (3, 32, 32)
    After that each patch is flattened into a vector of size 3 * 32 * 32 = 3072 , 3072-dimensional vector
    For each image, we got 5 * 7 * 7 = 245 patches of size 3072

    Parameters:
        image_list (torch.Tensor): Input tensor of shape (N, seq_len, C, H, W)
        n_patches (int): Number of patches to divide each image's dimension into.

    Returns:
        torch.Tensor: Patches of shape (N, seq_len * n_patches^2, C * (H/n_patches) * (W/n_patches)).
    """
    n, seq_len, c, h, w = image_list.shape

    assert h == w, "Patchify method is implemented for square images only"

    patch_size = h // n_patches

    image_list = image_list.view(n * seq_len, c, h, w)
    patches = image_list.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # patch extraction
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(n * seq_len, n_patches ** 2, -1)  # flatten
    patches = patches.view(n, seq_len * n_patches ** 2, -1)

    return patches


# ----------------------------------------------------------------------------------------------------------------------


def get_positional_encoding(seq_len, hidden_d):
    """Generate positional encoding for a given sequence length and hidden dimensionality.

    The positional encoding is computed using sine and cosine functions.
    It will be assigned to the input tokens to provide information about their position in the sequence.

    Parameters:
        seq_len (int): Length of the sequence.
        hidden_d (int): Dimensionality of the token embeddings.

    Returns:
        torch.Tensor: Positional encoding of shape (seq_len, hidden_d).
    """
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_d, 2, dtype=torch.float) * (-np.log(10000.0) / hidden_d))

    positional_encoding = torch.zeros((seq_len, hidden_d), dtype=torch.float)
    positional_encoding[:, 0::2] = torch.sin(position * div_term)
    positional_encoding[:, 1::2] = torch.cos(position * div_term)

    return positional_encoding


# ----------------------------------------------------------------------------------------------------------------------


class MHSA(nn.Module):
    """Multi-Head Self-Attention (MHSA) implementation for regression.

    Parameters:
        hidden_d (int): Dimensionality of token embeddings.
        n_heads (int): Number of attention heads.
    """
    def __init__(self, hidden_d, n_heads=2, device='cuda'):
        super(MHSA, self).__init__()

        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        assert self.hidden_d % self.n_heads == 0, f"Can't divide dimension {self.hidden_d} into {self.n_heads} heads"

        self.d_head = self.hidden_d // self.n_heads  # Dimensionality per head

        self.q_mappings = nn.Linear(self.hidden_d, self.hidden_d).to(self.device)
        self.k_mappings = nn.Linear(self.hidden_d, self.hidden_d).to(self.device)
        self.v_mappings = nn.Linear(self.hidden_d, self.hidden_d).to(self.device)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        """Forward pass for Multi-Head Self-Attention (MHSA).

        Parameters:
            sequences (torch.Tensor): Input tensor of shape (N, seq_len, D).

        Returns:
            torch.Tensor: Output tensor of shape (N, seq_len, D).
        """
        n_samples, seq_length, _ = sequences.shape

        q = self.q_mappings(sequences).view(n_samples, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_mappings(sequences).view(n_samples, seq_length, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_mappings(sequences).view(n_samples, seq_length, self.n_heads, self.d_head).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attention_probs = self.softmax(attention_scores)

        attention_output = torch.matmul(attention_probs, v)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(n_samples, seq_length, self.hidden_d)

        return attention_output


# ----------------------------------------------------------------------------------------------------------------------


class ViTBlock(nn.Module):
    """Encoder Block for Vision Transformer.

    Parameters:
        hidden_d (int): Dimensionality of each token.
        n_heads (int): Number of attention heads.
        mlp_ratio (int): Multiplier applied to hidden dimensionality for MLP.
    """
    def __init__(self, hidden_d, n_heads, mlp_ratio=4, device='cuda'):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.norm1 = nn.LayerNorm(self.hidden_d).to(self.device)

        self.mhsa = MHSA(self.hidden_d, self.n_heads).to(self.device)

        self.norm2 = nn.LayerNorm(self.hidden_d).to(self.device)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, mlp_ratio * self.hidden_d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_ratio * self.hidden_d, self.hidden_d),
            nn.Dropout(0.1)
        ).to(self.device)

    def forward(self, x):
        """Forward pass through the encoder block.

        Parameters:
            x (torch.Tensor): Input tensor of shape (N, seq_len, D)
              where N is batch size, seq_len is sequence length, D is token dimensionality.

        Returns:
            torch.Tensor: Normalized tokens of shape (N, seq_len, D)
        """
        normalized_tokens = self.norm1(x)

        mhsa_output = self.mhsa(normalized_tokens.to(self.device))

        res_con_out = x + mhsa_output

        normalized_tokens_two = self.norm2(res_con_out)

        res_con_out_two = res_con_out + self.mlp(normalized_tokens_two)

        return res_con_out_two


# ----------------------------------------------------------------------------------------------------------------------


class ViT(nn.Module):
    """Vision Transformer (ViT) implementation for regression tasks.

    Parameters:
        chw (tuple): Input shape (C, H, W) where C is number of channels, H is height, W is width.
        n_patches (int): Number of patches to divide each image's dimension into.
        n_blocks (int): Number of transformer blocks.
        hidden_d (int): Dimensionality of each token.
        n_heads (int): Number of attention heads.
        out_d (int): Output dimensionality.
    """
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=512, n_heads=2, out_d=1, device='cuda'):
        super(ViT, self).__init__()

        # Hyperparameters
        self.chw = chw  # ( Channels , Height , Width )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        assert (
                self.chw[1] % self.n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
                self.chw[2] % self.n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (self.chw[1] // self.n_patches, self.chw[2] // self.n_patches)

        self.patch_dim = int(self.chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.patch_dim, self.hidden_d).to(self.device)

        self.class_token = nn.Parameter(torch.randn(1, 1, self.hidden_d, device=self.device))

        self.blocks = nn.ModuleList(
            [ViTBlock(self.hidden_d, self.n_heads).to(self.device) for _ in range(self.n_blocks)]
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d)
        ).to(self.device)

    def forward(self, image_list):
        """Forward pass through the Vision Transformer.

        Parameters:
            image_list (torch.Tensor): Input tensor of shape (N, S, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_d) where N is batch size, out_d is the output dimensionality.
        """
        n, _, _, _, _ = image_list.shape

        patches = patchify(image_list, self.n_patches).to(self.device)  # torch.Size([N, 245, 512])

        patch_tokens = self.linear_mapper(patches)  # torch.Size([N, 245, 512])

        class_token = self.class_token.expand(n, 1, -1)
        tokens = torch.cat((class_token, patch_tokens), dim=1).to(
            self.device)  # torch.Size([N, 246, 512])

        pos_encoding = get_positional_encoding(tokens.size(1), self.hidden_d).to(self.device)
        tokens += pos_encoding  # torch.Size([N, 246, 512])

        for block in self.blocks:
            tokens = block(tokens)  # torch.Size([128, 246, 512])

        token = tokens[:, 0]  # Getting the classification token only

        return self.mlp(token)


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
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    train_outputs, train_labels = [], []
    for image_list, label_list in train_loader:
        image_list, label_list = image_list.to(device), label_list.to(device)

        optimizer.zero_grad()  # zero the gradients

        outputs = model(image_list)  # forward pass
        loss = criterion(outputs, label_list)  # computing loss

        loss.backward()  # backward pass
        optimizer.step()  # update weights

        running_loss += loss.item()
        train_outputs.append(outputs.detach())
        train_labels.append(label_list)

    avg_train_loss = running_loss / len(train_loader)

    train_outputs = torch.cat(train_outputs, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    rmse, mae, mape, r2 = compute_metrics(train_outputs, train_labels)

    return avg_train_loss, rmse, mae, mape, r2


def evaluate(model, loader, criterion, device):
    """Evaluate the model on the validation/test set."""
    model.eval()
    running_loss = 0.0
    valid_outputs, valid_labels = [], []
    with torch.no_grad():
        for image_list, label_list in loader:
            image_list, label_list = image_list.to(device), label_list.to(device)

            outputs = model(image_list)  # forward pass
            loss = criterion(outputs, label_list)  # computing loss

            running_loss += loss.item()
            valid_outputs.append(outputs)
            valid_labels.append(label_list)

    avg_valid_loss = running_loss / len(loader)

    valid_outputs = torch.cat(valid_outputs, dim=0)
    valid_labels = torch.cat(valid_labels, dim=0)

    rmse, mae, mape, r2 = compute_metrics(valid_outputs, valid_labels)

    return avg_valid_loss, rmse, mae, mape, r2


def main():
    # Hyperparameters
    n_epochs = 200
    learning_rate = 0.0005
    weight_decay = 0.0001
    batch_size = 32
    patience = 10
    n_patches = 7
    n_blocks = 8
    hidden_d = 256
    n_heads = 2

    train_dataset = ImageDataset(train_im, train_y)
    valid_dataset = ImageDataset(valid_im, valid_y)
    test_dataset = ImageDataset(test_im, test_y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    model = ViT((3, 224, 224), n_patches=n_patches, n_blocks=n_blocks, hidden_d=hidden_d, n_heads=n_heads, out_d=1).to(
        device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = MSELoss()

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    # Early stopping setup
    best_valid_loss = np.inf
    patience_counter = 0

    train_loss_list, valid_loss_list = [], []
    train_rmse_list, valid_rmse_list = [], []
    train_mae_list, valid_mae_list = [], []
    train_mape_list, valid_mape_list = [], []
    train_r2_list, valid_r2_list = [], []

    for epoch in range(n_epochs):
        train_loss, train_rmse, train_mae, train_mape, train_r2 = train(model, train_loader, criterion, optimizer,
                                                                        device)
        train_loss_list.append(train_loss)
        train_rmse_list.append(train_rmse)
        train_mae_list.append(train_mae)
        train_mape_list.append(train_mape)
        train_r2_list.append(train_r2)
        print(
            f"Epoch [{epoch + 1}/{n_epochs}] Train Loss: {train_loss:.4f} RMSE: {train_rmse:.4f} MAE: {train_mae:.4f} %MAE: {train_mape:.4f} R²: {train_r2:.4f}")

        valid_loss, valid_rmse, valid_mae, valid_mape, valid_r2 = evaluate(model, val_loader, criterion, device)
        valid_loss_list.append(valid_loss)
        valid_rmse_list.append(valid_rmse)
        valid_mae_list.append(valid_mae)
        valid_mape_list.append(valid_mape)
        valid_r2_list.append(valid_r2)
        print(
            f"Epoch [{epoch + 1}/{n_epochs}] Validation Loss: {valid_loss:.4f} RMSE: {valid_rmse:.4f} MAE: {valid_mae:.4f} %MAE: {valid_mape:.4f} R²: {valid_r2:.4f}")

        # Early stopping check
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    test_loss, test_rmse, test_mae, test_mape, test_r2 = evaluate(model, test_loader, criterion, device)
    print(
        f"Test Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, %MAPE: {test_mape:.4f}, R²: {test_r2:.4f}")

    # Plot metrics
    plot_filepath = f'../plots/plot_{t.localtime().tm_year}-{t.localtime().tm_mon}-{t.localtime().tm_mday}_{t.localtime().tm_hour}-{t.localtime().tm_min}-{t.localtime().tm_sec}.png'
    plt.figure(figsize=(12, 10))

    # Plot MAPE (Replaces MSE Plot)
    plt.subplot(2, 2, 1)
    plt.plot(train_mape_list, label='Trénovacia %MAE', color='blue', linestyle='-')
    plt.plot(valid_mape_list, label='Validačná %MAE', color='orange', linestyle='--')
    plt.xlabel('Epochy')
    plt.ylabel('MAE (%)')
    plt.title('Vývoj %MAE počas trénovania')
    plt.legend()
    plt.grid()

    # Plot RMSE
    plt.subplot(2, 2, 2)
    plt.plot(train_rmse_list, label='Training RMSE', color='green', linestyle='-')
    plt.plot(valid_rmse_list, label='Validation RMSE', color='red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('RMSE Development During Training')
    plt.legend()
    plt.grid()

    # Plot MAE
    plt.subplot(2, 2, 3)
    plt.plot(train_mae_list, label='Training MAE', color='purple', linestyle='-')
    plt.plot(valid_mae_list, label='Validation MAE', color='brown', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('MAE Development During Training')
    plt.legend()
    plt.grid()

    # Plot R²
    plt.subplot(2, 2, 4)
    plt.plot(train_r2_list, label='Training R²', color='cyan', linestyle='-')
    plt.plot(valid_r2_list, label='Validation R²', color='magenta', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.title('R² Development During Training')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(plot_filepath, dpi=300)
    plt.show()


main()
