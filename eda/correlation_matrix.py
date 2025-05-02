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
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

np.random.seed(0)
torch.manual_seed(0)


MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
MONTHS_WITHOUT_WINTER = ['03', '04', '05', '06', '07', '08', '09', '10']
# LOCATIONS = ['Alpnach']
LOCATIONS = {
    'Alpnach': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
    'Bern1': ['01', '02', '03', '04', '05', '06'],
    'Bern2': ['01', '02'],
    'NE': ['05', '06', '07', '08', '09']
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
    all_data = []

    for location, months in LOCATIONS.items():
        for month in months:
            csv_path = f'../datasets/tsi_dataset/expo{EXPO}_{location}{YEAR}/{month}_{YEAR}_complete_exposure{EXPO}/{month}_{YEAR}_expo{EXPO}_resized.csv'

            try:
                dataset = pd.read_csv(csv_path)
            except FileNotFoundError:
                print(f"Error: CSV file not found at {csv_path}")
                continue

            dataset = dataset.rename(columns={
                'Zenith': 'Zenit',
                'Temperature': 'Teplota vzduchu',
                'Humidity': 'Vlhkosť vzduchu',
                'Pressure': 'Tlak',
                'Irradiance': 'Solárne žiarenie'
            })
            all_data.append(dataset[['Zenit', 'Teplota vzduchu', 'Vlhkosť vzduchu', 'Tlak', 'Solárne žiarenie']])


    if all_data:
        all_data = pd.concat(all_data, ignore_index=True)
    else:
        all_data = None

    return all_data


df = read_data()

# ----------------------------------------------------------------------------------------------------------------------


# df.drop(columns=['Pressure'], inplace=True)

# df = df[df['Solárne žiarenie'] < 1000]
# df = df.reset_index(drop=True)
# df = df[df['Solárne žiarenie'] > 60]
# df = df.reset_index(drop=True)

# df = df[df['Temperature'] > 0]
# df = df.reset_index(drop=True)
#
# df = df[df['Pressure'] > 945]
# df = df.reset_index(drop=True)
# df = df[df['Pressure'] < 1080]
# df = df.reset_index(drop=True)

print(df)

# ----------------------------------------------------------------------------------------------------------------------


plt.figure(figsize=(8, 6))
plt.hist(df["Tlak"], bins=30, edgecolor="black", alpha=0.7, color="red")

plt.xlabel("Tlak", fontsize=14)
plt.ylabel("Frekvencia", fontsize=14)
plt.title("e)", fontsize=40)
plt.grid(True)

plt.savefig(f'../eda/hist_pressure_{t.localtime().tm_year}-{t.localtime().tm_mon}-{t.localtime().tm_mday}_{t.localtime().tm_hour}-{t.localtime().tm_min}-{t.localtime().tm_sec}.png', dpi=300)
plt.show()

# correlation_matrix = df.corr()
# plt.figure(figsize=(15, 13))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 15}, vmin=-1, vmax=1)
# plt.title("")
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.savefig(f'../eda/corr_matrix_{t.localtime().tm_year}-{t.localtime().tm_mon}-{t.localtime().tm_mday}_{t.localtime().tm_hour}-{t.localtime().tm_min}-{t.localtime().tm_sec}.png', dpi=300)
# plt.show()
