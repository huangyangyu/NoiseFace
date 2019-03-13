#!/usr/bin/env python
#coding: utf-8

import json
import copy
import random
import collections

#noise_ratio = 0.20
noise_ratio = 0.40
#noise_ratio = 0.60

flip_outlier_ratio = 0.5

def flip_noise():
    items = list()
    for line in open("ori_data.txt"):
        item = json.loads(line)
        item["ori_id"] = copy.deepcopy(item["id"])
        item["clean"] = True
        items.append(item)
    random.shuffle(items)

    m = int(round(noise_ratio * len(items)))
    ids = map(lambda item: item["id"][0], items[:m])
    random.shuffle(ids)
    for item, id in zip(items[:m], ids):
        if item["id"][0] != id:
            item["clean"] = False
            item["id"][0] = id
    for item in items[:m]:
        while item["clean"]:
            _item = random.choice(items[:m])
            if item["id"][0] != _item["ori_id"][0] and \
               item["ori_id"][0] != _item["id"][0]:
                id = item["id"][0]
                item["id"][0] = _item["id"][0]
                _item["id"][0] = id
                item["clean"] = _item["clean"] = False
    random.shuffle(items)
    for item in items:
        if item["clean"]:
            assert item["ori_id"][0] == item["id"][0]
        else:
            assert item["ori_id"][0] != item["id"][0]

    clean_data_file = open("clean_data.txt", "wb")
    flip_noise_data_file = open("flip_noise_data.txt", "wb")
    for item in items:
        if item["clean"]:
            clean_data_file.write(json.dumps(item) + "\n")
        else:
            flip_noise_data_file.write(json.dumps(item) + "\n")
    clean_data_file.close()
    flip_noise_data_file.close()


def add_outlier_noise():
    # clean: (1 - noise_ratio)
    items = list()
    for line in open("clean_data.txt"):
        item = json.loads(line)
        items.append(item)

    # flip noise: flip_outlier_ratio * noise_ratio
    flip_noise_items = list()
    for line in open("flip_noise_data.txt"):
        item = json.loads(line)
        flip_noise_items.append(item)
    flip_num = int(len(flip_noise_items) * flip_outlier_ratio)
    assert len(flip_noise_items) >= flip_num
    random.shuffle(flip_noise_items)
    items.extend(flip_noise_items[:flip_num])
    flip_noise_items = flip_noise_items[flip_num:]

    # outlier noise: (1 - flip_outlier_ratio) * noise_ratio
    outlier_num = len(flip_noise_items)
    outlier_noise_items = list()
    for line in open("outlier_noise_data.txt"):
        item = json.loads(line)
        outlier_noise_items.append(item)
    assert len(outlier_noise_items) >= outlier_num
    random.shuffle(outlier_noise_items)
    for i in xrange(outlier_num):
        item = outlier_noise_items[i]
        item["id"] = flip_noise_items[i]["id"]
        items.append(item)

    # output
    random.shuffle(items)
    with open("data.txt", "wb") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    """
    input:
    orignal dataset: ori_data.txt
    megaface 1m dataset: outlier_noise_data.txt
    output:
    data.txt
    """
    flip_noise()
    add_outlier_noise()

