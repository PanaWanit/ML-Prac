from time import sleep
from tqdm.auto import tqdm

for i in tqdm(range(10), desc='epoch', leave=False):
    for j in tqdm(range(10),  desc='batch', leave=False):
        sleep(.1)
    # print(i, end=' ')