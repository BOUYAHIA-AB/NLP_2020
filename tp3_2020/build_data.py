# -*- coding: utf-8 -*-
import csv
import os
import sys

science_dataset = {
    'train' : "./tp3_data/science/train.txt",
    'dev' : "./tp3_data/science/dev.txt",
    'test' : "./tp3_data/science/test.txt",
}
disease_dataset = {
    'train' : "./tp3_data/disease/train.txt",
    'dev' : "./tp3_data/disease/dev.txt",
    'test' : "./tp3_data/disease/test.txt",
}

def load_dataset(filename):
    """Loads dataset into memory from csv file"""
    with open(filename, 'r') as fp:
        dataset = []
        words, tags = [], []
        # Each line of the file corresponds to one word and it's tag
        for line in fp :
            line_split = line.split()
            if len(line_split) == 0 :
                continue
            try:
                word, tag = str(line_split[0]), str(line_split[1])
                if tag[0] == 'B':
                    tag = 'I' + tag[1:]    
                words.append(word)
                tags.append(tag)
            except UnicodeDecodeError as e:
                print("An exception was raised, skipping a word: {}".format(e))
                pass
            # If dot is encontred it means the sentence is finished
            if line_split[0] == ".":
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []

    print("Number of sentences : " , len(dataset))
    return dataset

def save_dataset(dataset, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
        with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
            for words, tags in dataset:
                file_sentences.write("{}\n".format(" ".join(words)))
                file_labels.write("{}\n".format(" ".join(tags)))
    print("- done.")

if __name__ == '__main__':

    for key, value in science_dataset.items() :
        # Load the dataset into memory
        print("Loading dataset into memory...")
        dataset = load_dataset(value)
        print("- done.")
        # Save the datasets to files
        save_dataset(dataset, 'data/science/'+key)

    for key, value in disease_dataset.items() :
        # Load the dataset into memory
        print("Loading dataset into memory...")
        dataset = load_dataset(value)
        print("- done.")
        # Save the datasets to files
        save_dataset(dataset, 'data/disease/'+key)

    


