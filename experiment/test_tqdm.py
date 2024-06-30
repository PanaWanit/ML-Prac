from time import sleep
from tqdm.auto import tqdm

for i in tqdm(range(10)):
    for j in tqdm(range(10), leave=False):
        sleep(.1)
    # print(i, end=' ')