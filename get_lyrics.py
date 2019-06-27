from os import listdir
from urllib import request
from bs4 import BeautifulSoup
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding


SONG_URLS_FILENAME = 'songs.txt'
LYRICS_DIR = 'lyrics/'
ALL_LYRICS_FILENAME = 'all_lyrics.txt'

# sigmoid function
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def clean_div_content(div_content):
    retval = str(div_content)
    retval = retval.replace('<p>', '\n')
    retval = retval.replace('<br/>', '\n')
    retval = retval.replace('</p>', '')
    retval = retval.replace('</div>', '')
    retval = retval.replace('<div class="cnt-letra p402_premium">', '')
    return retval.strip()

def download_songs():
    url_file = open(SONG_URLS_FILENAME, 'r')
    urls = url_file.readlines()
    url_file.close()
    index = 0
    for url in urls:
        print('Retriving ' + url)
        lyric_filename = LYRICS_DIR + str(index)
        response = request.urlopen(url.strip())
        soup = BeautifulSoup(response, 'lxml')
        content = soup.findAll('div', attrs={'class':'cnt-letra'})[0]
        lyric_file = open(lyric_filename, 'w')
        lyric_file.write(clean_div_content(content))
        lyric_file.close()
        index = index + 1
        print('OK')

def merge_files():
    all_content = ''
    for filename in listdir(LYRICS_DIR):
        file = open(LYRICS_DIR + filename, 'r')
        all_content += file.read().replace('\t', '')
        file.close()
    output_file = open(ALL_LYRICS_FILENAME, 'w')
    output_file.write(all_content.lower())
    output_file.close()

def process():
    # reads the text
    file = open(ALL_LYRICS_FILENAME, 'r')
    text = file.read()
    file.close()

    # creates a set of all unique chars found in the text
    all_chars = sorted((list(set(text))))

    # creates a dictionary to map each char to a number
    char_indices = dict((c, i) for i, c in enumerate(all_chars))

    # creates a dictionary to map back each number to a char
    indices_char = dict((i,c) for i, c in enumerate(all_chars))

    available_input = []
    expected_output = []
    sequence_length = 20
    for i in range(0, len(text) - sequence_length):
        available_input.append(text[i: i + sequence_length])
        expected_output.append(text[i + sequence_length])

    # create empty matrices for input and output sets
    # x = np.zeros((len(available_input), sequence_length, len(all_chars)), dtype=np.bool)
    # y = np.zeros((len(expected_output), len(all_chars)), dtype=np.bool)
    x = np.zeros(len(available_input), sequence_length, dtype=np.int8)
    y = np.zeros(sequence_length, dtype=np.int8)

    # converts each char to its related index and add them into the matrices
    #for i, inpt in enumerate(available_input):
    #    for t, char in enumerate(inpt):
    #        x[i, t, char_indices[char]] = 1
    #    y[i, char_indices[expected_output[i]]] = 1

    model = Sequential()

    # A word embedding is a class of approaches for representing words and documents using a dense vector
    # representation. Keras offers an Embedding layer that can be used for neural networks on text data.
    #
    # input_dim: This is the size of the vocabulary in the text data. For example, if your data is integer
    #            encoded to values between 0-10, then the size of the vocabulary would be 11 words.
    # output_dim: This is the size of the vector space in which words will be embedded. It defines the size
    #            of the output vectors from this layer for each word. For example, it could be 32 or 100 or even
    #            larger. Test different values for your problem.
    # input_length: This is the length of input sequences, as you would define for any input layer of a Keras model.
    #            For example, if all of your input documents are comprised of 1000 words, this would be 1000.
    model.add(Embedding(len(all_chars), 50, input_length=len(available_input)))

    # model.add(Dense(len(all_chars), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x, y, batch_size=128, epochs=30)

# download_songs()
# merge_files()
process()

