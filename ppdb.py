import sys
from nltk import word_tokenize
import string
from nltk.stem import SnowballStemmer


PPDB_DB = "./rep_ppdb/ppdb-2.0-tldr"

############ THIS IS THE NLTK-BASED PARAPHRASE GENERATOR !!!! ##########
def add_to_dict_of_set(key, value, dict_set):
    if key in dict_set:
        dict_set[key].add(value)
    else:
        dict_set[key] = {value}


def clean_paraphrase(paraphrase_dict):
    stemmer = SnowballStemmer("english")
    paraphrase_dict_clean = dict()
    print("Size: %d" % len(paraphrase_dict))

    for phrase, paraphrases in paraphrase_dict.items():
        new_paraphrases = set()
        for paraphrase in paraphrases:
            if stemmer.stem(phrase) != stemmer.stem(paraphrase):
                new_paraphrases.add(paraphrase)
        if len(new_paraphrases):
            paraphrase_dict_clean[phrase] = new_paraphrases
    print("Size: %d" % len(paraphrase_dict_clean))
    return paraphrase_dict_clean


############ THIS IS THE PPDB-BASED PARAPHRASE GENERATOR !!!! ###########
def string_clean(words):
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    return stripped


############ THIS IS THE PPDB-BASED PARAPHRASE GENERATOR !!!! ###########
class PPDB_dict(object):
    def __init__(self, ppdb_file):
        self.ppdb_dict = dict()

        with open(ppdb_file, "r") as ppdb_f:
            lines = ppdb_f.readlines()
            for line in lines:
                baseword = line.split("|||")[1].strip()
                ppword = line.split("|||")[2].strip()
                score = line.split("|||")[3].split(" ")[1].split("=")[1]

                if (line.split("|||")[-1].strip() == "Equivalence") \
                        or (line.split("|||")[-1].strip() == "ForwardEntailment") \
                        or (line.split("|||")[-1].strip() == "ReverseEntailment"):
                    self.add_paraphrases(baseword, ppword, score)


        self.ppdb_dict = {k: v for k, v in sorted(self.ppdb_dict.items(), key=lambda item: item[1])}


    def add_paraphrases(self, baseword, ppword, score):
        if baseword == ppword:      # don't add a self-paraphrase
            return
        if self.ppdb_dict.get(baseword):
            if ppword in self.ppdb_dict.get(baseword, []):   # don't add the same thing again
                return
            self.ppdb_dict[baseword].append((ppword,score))
            # self.ppdb_paraphrases[baseword] += [ppword, score]
        else:
            self.ppdb_dict[baseword] = [(ppword, score)]



class PPDB(object):
    def __init__(self, sentence, ppdb_dict, isTxt = True):
        self.sentence = sentence
        self.ppdb_dict = ppdb_dict
        self.ppdb_paraphrases = dict()
        self.tokens = []
        self.token_paraphrases = dict()
        if isTxt:
            with open(sentence, "r") as f_sentence:
                lines = f_sentence.readlines()
                for line in lines:
                    tokens = word_tokenize(line)
                #tokens = [string_clean(x.split()) for x in words]
                self.tokens = tokens
        else:
            tokens = word_tokenize(sentence)
            self.tokens = tokens


    def save_ppdb(self, filename, sortDict = True):
        if sortDict:
            self.sort_paraphrases()
        #TODO: fix this part if we need to save the ppdb results
        with open(filename, "w") as f_save:
            n = 0
            for token in self.tokens:
                if token == "UNK":
                    write_line = "</s> </s>\n"
                elif token in self.ppdb_paraphrases.keys():
                    write_line = str(token) + " "
                    for ppword in self.ppdb_paraphrases[token]:
                        write_line += ppword + " "
                        write_line += "</s>\n"
                else:
                    write_line = str(token) + " </s>\n"
                    f_save.write(write_line)
                    n += 1
                    if n % 1000 == 0:
                        f_save.flush()


    # def get_token_paraphrases(self):
    #     for t in self.tokens:
    #         # self.token_paraphrases[t] = []
    #         if t in self.ppdb_dict:
    #             self.token_paraphrases[t] = self.ppdb_dict[t]
    #
    #     return self.token_paraphrases
    #
    def get_n_paraphrases(self, num_paraphrases):
        # self.sort_paraphrases()
        n_paraphrases = {}
        for token in self.tokens:
            if token in self.ppdb_dict:
                for i in range(num_paraphrases):
                    if token in n_paraphrases.keys():   # in case repeated tokens in sentenc
                        try:
                            n_paraphrases[token].add(self.ppdb_paraphrases[token][i][0])
                        except:
                            print("less than n paraphrases!!!")
                    else:
                        try:
                            n_paraphrases[token] = {self.ppdb_paraphrases[token][i][0]}
                        except:
                            print("less than n paraphrases!!! ")

        return n_paraphrases

    def gen_paraphrase_questions(self, num_replacement, num_paraphrases):

        n_paraphrases = self.get_n_paraphrases(num_paraphrases)
        paraphrases = {}

        cnt = 0
        i = 0
        while cnt <= num_replacement:
            for idx, token in enumerate(self.tokens):
                p = 0
                if token in n_paraphrases.keys():
                    while p < len(n_paraphrases[token]):
                        tokens = list(self.tokens)
                        tokens[idx] = list(n_paraphrases[token])[p]
                        paraphrases[i] = tokens
                        p += 1
                        i += 1
                    cnt += 1

        paraphrase_list = []
        for id, sentence in paraphrases.items():
            paraphrase_list.append(" ".join(sentence))
        return paraphrase_list


if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     vocab = sys.argv[1]
    #     lexicon = sys.argv[2]
    # else:
    #     vocab = "vocab2.txt"
    #     lexicon = "ppdb-2.0-tldr"
    # ppdb = PPDBE(vocab, lexicon, isTxt=True)
    # ppdb.read_lexicon()
    # ppdb.save_ppdb()

    ppdb = PPDB_dict(PPDB_DB)
    phrase = "this is an example sentence to paraphrase"

    new_ppdb_generator = PPDB(phrase, ppdb, isTxt = False)
    new_ppdb_generator.gen_paraphrase_questions(2,3)
    #print(ppdb.gen_paraphrase_questions(2, 3))

