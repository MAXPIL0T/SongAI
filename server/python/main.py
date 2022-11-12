# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import csv
"""
import string
import numpy as np
import nltk
import pandas as pd
"""

if __name__ == "__main__":
    artist_to_genre = dict()
    lyrics_to_artist = dict()
    lyrics_to_genre = dict()

    with open('artists-data.csv') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in file:
            if row[0] not in artist_to_genre:
                artist_to_genre.update({row[4]: [elm.lstrip() for elm in row[1].split(';')]})

    # print(artist_to_genre)

    with open('lyrics-data.csv') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='"')
        cnt = 0
        for row in file[1:]:
            if row[-2] == 'SLink':
                continue
            artist = row[2]
            if artist not in lyrics_to_artist:
                lyrics_to_artist.update({row[-2]: artist})

    for key, val in lyrics_to_artist.items():
        lyrics_to_genre.update({key: artist_to_genre[val]})

    # print(lyrics_to_artist)

    # print(lyrics_to_artist)
