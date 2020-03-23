from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from nltk import Tree, sent_tokenize
from pprint import pprint as pp
from typing import List, Tuple, Dict

from setup import word_tokenize


k_populate_replacement_list_with_no_replace_phrases = True


# todo: config should be input
class AllenPredictor:
    """
    A class to use tree parsing to extract phrases of a certain type (see __init__ for details of
    the valid phrases (length and type)
    """
    def __init__(self, min_phrase_len=3):
        archive = load_archive("./rep_allennlp/elmo-constituency-parser-2018.03.14.tar.gz")
        self.predictor = Predictor.from_archive(archive, 'constituency-parser')

        self.min_phrase_len = min_phrase_len        # how many leaves when we stop

        self.replacement_node_types = ["-DNR", "NP", "VP", "VBZ", "VBN"]    # todo (should be input)
        self.num_replacement_node_types = len(self.replacement_node_types)

        self.tree_label_to_index = {name: i for i, name in enumerate(self.replacement_node_types)}

        print(f'Initialied allen tree predictor with stop leaves={self.min_phrase_len} '
              f'and the following substitution nodes')
        pp(self.replacement_node_types)

    def get_allen_tree(self, sentence : Dict[str,List[str]]) -> Tree:
        """
        Returns a parse tree from the loaded allen module
        :param sentence: Expects dict of the form: {"sentence": sentence}
        :return: a Tree
        """
        output = self.predictor.predict_json(sentence)
        return Tree.fromstring(output["trees"])

    # parse the tree for a question into the phrasal replacement candidates
    # Note that tree is **modified** (all leaves will have an appended "_index"
    def sentence_to_phrases(self, sentence, include_non_matching_phrases=False) -> Tuple[List[Dict], List[Dict]]:
        """
        Parses a tree into the set of phrases that meet certain criteria.
        :param sentence: The input sentence (list of strings)
        :param include_non_matching_phrases: whether to insert the phrases that don't match the rules with
        label == 0 for Do Not Replace
        This should always be false when parsing a context

        :return: tuple:
            - list of entries for each phrase, each entry: phrase, phrase_type_numeric, spans (tuple)
            - list of entries for each non-replace phrase, each entry: phrase, spans
            dict: phrase: phrase_text
                phrase_type: numeric
                spans: tuple(start, end)
        """

        # Modifies a tree ** be careful **
        def _add_indices_to_terminals(t):
            for idx, _ in enumerate(t.leaves()):
                tree_location = t.leaf_treeposition(idx)
                non_terminal = t[tree_location[:-1]]
                non_terminal[0] = non_terminal[0] + "_" + str(idx)
            return t

        def _internal_traverse(t):
            for child in t:
                if not isinstance(child, Tree):  # early fail if it's not a tree
                    return

                label = child.label()
                num_leaves = len(child.leaves())
                # If haven't met stop criterion (phrase type / length), recurse
                if num_leaves > self.min_phrase_len or label not in self.tree_label_to_index:
                    _internal_traverse(child)
                else:  # valid stop (few enough leaves, and it's in the replacement list)
                    phrase = child.leaves()
                    nonlocal phrases_to_paraphrase
                    phrases_to_paraphrase += [(phrase, self.tree_label_to_index[label])]  # phrase, phrase type (numeric)

        # Preprocessing
        tree = allen_predictor.get_allen_tree({"sentence": sentence})
        orig_sentence = tree.leaves()       # must do before calling the below (since tree will be modified)

        # Traverse the tree and extract the phrases that are candidates for replacement (paraphrasing)
        tree_with_leaf_indices = _add_indices_to_terminals(tree)   # add indices to tree roots
        phrases_to_paraphrase = []        # accumulation set from tree traversal List[(phrase, type_of_phrase_numeric)]
        _internal_traverse(tree_with_leaf_indices)   # modifies phrases_to_paraphrase

        # Do final accumulation and insert any pieces that did not match our criteria
        output_list = []                                # final accumulation (postprocess - remove the numbers)
        non_replacement_list = []                       # also track all the parts of the question we don't p-phrase
        last_unused = 0                                 # index of last un-"used" word in the contextjj

        for phrase_words, label_numeric in phrases_to_paraphrase:  # need to remove the indices that were appended
            phrase_start_idx = int(phrase_words[0].split("_")[1])  # index of the first word in this phrase
            for i, word in enumerate(phrase_words):                # iterate over the words in each phrase, p
                phrase_words[i] = word.split("_")[0]
            phrase_len = len(phrase_words)

            # also grab the unused portions (do this before we add the next actual phrase)
            if phrase_start_idx > last_unused:      # we have some part that was skipped
                # this appends a phrase of arbitrary length
                non_replacement_list += [{"phrase": orig_sentence[last_unused:phrase_start_idx],
                                          "span": (last_unused, phrase_start_idx - 1)}]

                # This adds a phrase of max_len = max_phrase_len to the set of replacement phrase candidates
                # This avoids special manipulation in the model
                if include_non_matching_phrases:
                    while phrase_start_idx > last_unused:
                        curr_len = min(self.min_phrase_len, phrase_start_idx - last_unused)
                        output_list += [{"phrase": orig_sentence[last_unused: last_unused + curr_len],
                                         "type": 0,  # the DNR index
                                         "span": (last_unused, last_unused + curr_len - 1)}]  # (start, end) of phrase
                        last_unused += curr_len

            last_unused = phrase_start_idx + phrase_len

            # all intermediate, un-used parts have been added, so now add the next piece
            output_list += [{"phrase": phrase_words,               # phrase_text
                             "type": label_numeric,                # type of phrase (numeric)
                             "span": (phrase_start_idx, phrase_start_idx + len(phrase_words) - 1)}]  # (start, end)
        return output_list, non_replacement_list

    def context_to_replacement_phrase_sets(self, context_paragraph) -> List[List[str]]:
        """
        :param context_paragraph: a list of strings (each string a sentence)
        :return: List (indexed by type of phrase) of List of phrases (type of phrase, int => list of phrases)
        output[i] is the set of phrases for label i type
        """

        # Use dictionaries here to avoid multiple copies of the same phrase. Then throw away the key since not needed
        phrase_type_to_phrases = [dict() for _ in range(self.num_replacement_node_types)]

        context_sentences = sent_tokenize(context_paragraph)
        phrases_for_each_sentence = []
        for idx, sent in enumerate(context_sentences):
            phrases_for_each_sentence.append(self.sentence_to_phrases(sent, include_non_matching_phrases=False)[0])

        start_index_for_sentence = 0
        for idx, phrase_list in enumerate(phrases_for_each_sentence):
            for phrase in phrase_list:
                phrase_joined = " ".join(phrase["phrase"])
                numerical_label = phrase["type"]
                if phrase_type_to_phrases[numerical_label].get(phrase_joined):   # only add given phrase once
                    continue

                # Adjust the indices to reflect index in the overall context
                old_start, old_end = phrase["span"]
                phrase["span"] = (old_start + start_index_for_sentence, old_end + start_index_for_sentence)
                phrase_type_to_phrases[numerical_label][phrase_joined] = phrase  # store to dict

            # update the start spot for next iteration
            sentence_length_for_idx = len(word_tokenize(context_sentences[idx]))
            start_index_for_sentence += sentence_length_for_idx

        # throw away the keys and keep values (the separated phrases)
        return [list(x.values()) for x in phrase_type_to_phrases]    # (list maps idx => phrase)


if __name__ == "__main__":
    # allen setup
    allen_predictor = AllenPredictor()

    # setup of two examples
    in_sentence = "This is a sentence to be predicted from a sentence"
    in_context_paragraph = "This is a context paragraph about coding in python. Python is great. It has really cool typing support."

    # example of direct string parse to save compute time on allen predictor
    # tree_string = ('(S (NP (DT This)) (VP (VBZ is) (NP (NP (DT a) (NN sentence)) (SBAR (S (VP (TO to) (VP (VB be) (VP (VBN predicted)))))))) (. !))')
    # tree = Tree.fromstring(tree_string)
    # pp(tree_string)

    # Actual run - parse question and context
    replace, nonreplace = allen_predictor.sentence_to_phrases(in_sentence, include_non_matching_phrases=k_populate_replacement_list_with_no_replace_phrases)
    pp(replace)
    pp(nonreplace)
    pp(allen_predictor.context_to_replacement_phrase_sets(in_context_paragraph))
