from os import listdir
from urllib import request
from bs4 import BeautifulSoup
import numpy as np

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
    x = np.zeros((len(available_input), sequence_length, len(all_chars)), dtype=np.bool)
    y = np.zeros((len(expected_output), len(all_chars)), dtype=np.bool)

    # converts each char to its related index and add them into the matrices
    for i, inpt in enumerate(available_input):
        for t, char in enumerate(inpt):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[expected_output[i]]] = 1

    np.random.seed(1)

    x = x[:10]
    y = y[:10]

    # synapse 0, first layer of weights. Initializing it with random weights
    syn0 = 2*np.random.random( ( len(available_input), sequence_length, len(all_chars) ) ) - 1
    # syn0 = 2*np.random.random((10, sequence_length, len(all_chars))) - 1

    for iter in range(10000):
        print(iter)
        # forward propagation
        l0 = x
        l1 = nonlin(np.dot(l0, syn0))

        # calculates the error, which is the difference of the value to the correct answer
        l1_error = y - l1
        l1_delta = l1_error * nonlin(l1, True)
        
        # updates the weights
        syn0 += np.dot(l0.T, l1_delta)

    print("Trained.")
    # print(l1)

# download_songs()
# merge_files()
process()

