# def GetData(path, to_lower=False, count_threshold=None):
#     # if count_threshold is None then all words are kept, else
#     # the words with frequency < count_threshold are removed

#     # to_lower = True implies that corpus is converted to lower case

#     files = os.listdir(path)
#     data = []
#     for file in files:
#         with open(os.path.join(path, file), "r") as file:
#             file = file.readlines()
#             for line in file:
#                 line = nltk.word_tokenize(line)
#                 line = RemovePunctuations(line, to_lower=to_lower)
#                 data += line

#     data = ReplaceWithUnk(data, count_threshold=count_threshold)

#     vocabulary = {}
#     # vocabulary is word: index
#     inverse_vocabulary = {}
#     # inverse_vocabulary is index: word
#     index = 0
#     for word in data:
#         if(word in vocabulary):
#             vocabulary[word] += 1
#         else:
#             vocabulary[word] = index
#             inverse_vocabulary[index] = word
#             index += 1

#     return data, vocabulary, inverse_vocabulary
