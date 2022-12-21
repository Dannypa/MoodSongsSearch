import torch
import faiss
import os
from sentence_transformers import SentenceTransformer
from model import Net, DEVICE
import config 
import tqdm
import json
import numpy as np
import faiss
import time


def dict_to_list_fixed(d):
    tmp = list(d.items())
    tmp.sort()
    return [i[1] for i in tmp]


tname = "multi-qa-mpnet-base-dot-v1"
# tname = "paraphrase-multilingual-MiniLM-L12-v2"
bert = SentenceTransformer(tname)
print(os.getcwd())


# loading model
checkpoint = torch.load(os.path.join(config.SAVE_DIR, config.MODEL_NAME + "-" + tname + "-FINAL.pt"), map_location=DEVICE)
model = Net(number_params=5)
model.load_state_dict(checkpoint) 
# yes model


# loading data

# lyrics = []
# meta_lyr = []
# dir = config.LYRICS_PATH
# for file in tqdm.tqdm(os.listdir(dir)):
#     # if len(meta_lyr) > 100:
#     #     break
#     with open(os.path.join(dir, file), 'r') as f:
#         cur = json.load(f)
#         for el in cur:
#             lyrics.append(el['lyrics'])
#             meta_lyr.append(el['meta'])


song_data = []
meta = []
dir = config.META_PATH
for file in tqdm.tqdm(os.listdir(dir)):
    with open(os.path.join(dir, file), 'r') as f:
        # if len(meta) > 100:
        #     break   
        cur = json.load(f)
        # print(cur)
        for el in cur['data']:
            song_data.append((el['song'], el['artist']))
            lst = dict_to_list_fixed(el['meta'])
            # meta.append(np.array(lst))
            meta.append(np.array([lst[j] for j in [0, 1, 3, 5, 6]]))
            # print(meta)
            # print(meta * np.array(config.meta_max_val))
            # exit(0)


# print(meta)

# def squared_sum(a):
#     return np.sum(a ** 2)
#     return sum([el **2 for el in a])

# def acc(a, b):
#         assert len(a) == len(b)
#         # return np.sqrt(np.sum([
#         #     (a - b)**2
#         # ]))
#         numerator = sum(xa * xb for xa, xb in zip(a, b))
#         denominator = squared_sum(a) * squared_sum(b)
#         return (numerator / float(denominator))

# lyrics = np.array(lyrics)
# meta = np.array(meta)
# meta_lyr = np.array(meta_lyr)

# indexLyrics = faiss.IndexIDMap(faiss.IndexFlatIP(lyrics[0].shape[0]))
# indexLyrics.add_with_ids(lyrics, np.array(range(0, len(lyrics))))
# faiss.write_index(indexLyrics, "lyrics_model")

# indexMeta = faiss.IndexIDMap(faiss.IndexFlatIP(meta[0].shape[0]))
indexMeta = faiss.IndexFlatL2(meta[0].shape[0])
indexMeta.add(np.array(meta))
faiss.write_index(indexMeta, "meta_model")



def dist(v1, v2):
    return np.sqrt(
        np.sum(
            (v1 - v2) ** 2
        )
    ), np.sqrt(np.sum(v1**2))



def find_k_best(cur, lk:int, arr:np.array, f:callable):
    return indexMeta.search(np.array([cur]), lk)[1].tolist()[0], -1
    # mns = [1_000_000_000 for i in range(lk)]
    # inds = [-1 for i in range(lk)]
    # for i, val in enumerate(arr):
    #     d = f(val, cur)
    #     for j in range(lk):
    #         if inds[j] == -1 or mns[j] > d:
    #             for p in range(lk - 1, j, -1):
    #                 inds[p] = inds[p - 1]
    #                 mns[p] = mns[p - 1]
    #             inds[j] = i
    #             mns[j] = d
    #             break
    # return inds, mns


last = None
def search_meta(query):
    global last
    t = time.time()
    query_vector = model((torch.tensor(bert.encode([query])) / config.lyrics_max_val)).detach().numpy()[0] * config.meta_max_val
    mi, mns = find_k_best(query_vector, 5, meta, dist)
    print('totaltime: {}'.format(time.time()-t))
    return mi
    # print(query_vector)
    # lk = 1  # short for lyrics k
    # print(query)
    # # print(bert.encode(query))
    # cur = bert.encode(query)
    # # if not (last is None):
    # #     print(np.sqrt(np.sum((cur - last) ** 2)), np.sum(np.sqrt(cur ** 2)))
    # last = cur
    # # best_meta = indexLyrics.search(bert.encode([query]), lk)[1].tolist()[0]
    # # mns = [1_000_000_000 for i in range(lk)]
    # # inds = [-1 for i in range(lk)]
    # # for i, val in enumerate(lyrics):
    # #     d = dist(val, cur)
    # #     for j in range(lk):
    # #         if inds[j] == -1 or mns[j] > d:
    # #             for p in range(lk - 1, j, -1):
    # #                 inds[p] = inds[p - 1]
    # #                 mns[p] = mns[p - 1]
    # #             inds[j] = i
    # #             mns[j] = d
    # #             break
    # inds, mns = find_k_best(cur, lk, lyrics, dist)
    # print(inds)
    # print(mns)
    # eta = meta_lyr[inds[0]]
    # mi, mns = find_k_best(eta, 5, meta, dist)
    # return mi
    # k = 5
    # res = []
    # for mt in best_meta:
    #     print(dist(lyrics[mt], cur))
    #     # print(meta_lyr[mt])
    #     # print(np.array([meta_lyr[mt]]).shape)
    #     top_k = indexMeta.search(np.array([meta_lyr[mt]]), k)
    #     res.extend(top_k[1].tolist()[0])
    # print('totaltime: {}'.format(time.time()-t))
    # return res
    # # return top_k[1].tolist()[0]


print("Testing...")
cnt = dict()
with torch.no_grad():
    while True:
        q = input()
        for ind in search_meta(q):
            print(song_data[ind])
            if ind in cnt:
                cnt[ind] += 1
            else:
                cnt[ind] = 1
            # print(song_data[ind][0] + " - " + song_data[ind][1])
        print(sorted(cnt.values()))
        # for i in cnt:
        #     print(song_data[i], meta[i]) 
        