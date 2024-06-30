from time import sleep
from tqdm import tqdm

for i in tqdm(range(10), desc='epoch', leave=True):
    for j in tqdm(range(10),  desc='batch', leave=(i==9)):
        sleep(.025)
    # print(i, end=' ')