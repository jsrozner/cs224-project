from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from nltk import Tree, sent_tokenize
from pprint import pprint as pp
from typing import List, Dict, Tuple

from setup import word_tokenize


# todo: config should be input
class AllenPredictor():
    def __init__(self):
        archive = load_archive("./rep_allennlp/elmo-constituency-parser-2018.03.14.tar.gz")
        self.predictor = Predictor.from_archive(archive, 'constituency-parser')

        self.stop_leaves = 3
        self.replacement_node_types = ["NP", "VP", "VBZ", "VBN"] # todo

        self.num_replacement_node_types = len(self.replacement_node_types)
        self.tree_label_to_index = {name: i for i, name in enumerate(self.replacement_node_types)}

    def get_allen_tree(self, sentence) -> Tree:
        output = self.predictor.predict_json(sentence)
        #pp(output["trees"])
        return Tree.fromstring(output["trees"])


    # parse the tree for a question into the phrasal replacement candidates
    # Note that tree is **modified** (all leaves will have an appended "_index"
    def _replaceable_phrases_from_tree(self, start_tree: Tree) -> List[Dict]:
        """
        Parses a tree into the set of phrases that meet certain criteria.

        :param start_tree: The input tree representing a sentence. Parse looking for phrases of certain type
        :return: List (indexed by label type) => tuple of (words in phrase, len of phrase)
        """

        # Modifies a tree ** be careful **
        def _add_indices_to_terminals(tree):
            for idx, _ in enumerate(tree.leaves()):
                tree_location = tree.leaf_treeposition(idx)
                non_terminal = tree[tree_location[:-1]]
                non_terminal[0] = non_terminal[0] + "_" + str(idx)
            return tree

        def _internal_traverse(tree):
            for child in tree:
                if not isinstance(child, Tree): # early fail if it's not a tree
                    return

                label = child.label()
                num_leaves = len(child.leaves())
                if num_leaves > self.stop_leaves or label not in self.tree_label_to_index:            # haven't met stop criterion, so recurse
                    _internal_traverse(child)
                else:  # valid stop (few enough leaves, and it's in the replacement list)
                    phrase = child.leaves()
                    nonlocal phrases_to_paraphrase
                    phrases_to_paraphrase += [(phrase, self.tree_label_to_index[label])] # phrase, type of phrase

        tree_with_leaf_indices = _add_indices_to_terminals(start_tree)   # add indices to tree roots
        phrases_to_paraphrase = []                   # accumulation set from tree traversal
        _internal_traverse(tree_with_leaf_indices)   # modifies phrases_to_paraphrase

        output_list = []                                # final accumulation (postprocess)
        for p, label_numeric in phrases_to_paraphrase:  # each p is a tuple (phrase, label_numeric)
            phrase_start_idx = int(p[0].split("_")[1])
            for i, word in enumerate(p):                # iterate over the words in each phrase, p
                p[i] = word.split("_")[0]
            output_list += [{"phrase": p,
                             "type": label_numeric,
                             "span": (phrase_start_idx, phrase_start_idx + len(p) - 1)}]

        return output_list


    def _question_to_nonreplacement_portions(self, replacement_candidates, sentence):
        non_replacement_list = []
        last_unused = 0
        for p in replacement_candidates:
            phrase = p["phrase"]
            phrase_len = len(phrase)
            phrase_start = p["span"][0]
            if phrase_start > last_unused:
                non_replacement_list += [{"phrase": sentence[last_unused:phrase_start],
                                          "span": (last_unused, phrase_start - 1)}]
            last_unused = phrase_start + phrase_len

        return non_replacement_list


    def parse_question_tree_for_phrases(self, question) -> Tuple:
        tree = allen_predictor.get_allen_tree({"sentence": question})

        orig_sentence = tree.leaves()       # must do before calling the below (since tree will be modified)
        replacement_candidates = self._replaceable_phrases_from_tree(tree)
        non_replacement = self._question_to_nonreplacement_portions(replacement_candidates, orig_sentence)   # orig sentence must be generated prior to calling replacementcandidates

        return replacement_candidates, non_replacement


    def context_to_replacement_phrase_sets(self, context_paragraph) -> List[List[str]]:
        """
        :param context_paragraph: a list of strings (each string a sentence)
        :return: List (indexed by type of phrase) of List of phrases (type of phrase, int => list of phrases)
        """

        # Use dictionaries here to avoid multiple copies of the same phrase. Then throw away the key since not needed
        phrase_type_to_phrases = [dict() for i in range(self.num_replacement_node_types)]

        context_sentences = sent_tokenize(context_paragraph)
        phrases_for_each_sentence = []
        for idx, s in enumerate(context_sentences):
            tree = allen_predictor.get_allen_tree({"sentence": s})
            phrases_for_each_sentence.append(self._replaceable_phrases_from_tree(tree))

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
                phrase_type_to_phrases[numerical_label][phrase_joined] = phrase # store to dict

            # update the start spot for next iteration
            sentence_length_for_idx = len(word_tokenize(context_sentences[idx]))
            start_index_for_sentence += sentence_length_for_idx

        # throw away the keys and keep values (the separated phrases)
        return [list(x.values()) for x in phrase_type_to_phrases]    # (list maps idx => phrase)

if __name__ == "__main__":
    # allen setup
    allen_predictor = AllenPredictor()

    # setup of two examples
    sentence = "This is a sentence to be predicted from a sentence"
    context_paragraph = "This is a context paragraph about coding in python. Python is great. It has really cool typing support."

    # example of direct string parse to save compute time on allen predictor
    # tree_string = ('(S (NP (DT This)) (VP (VBZ is) (NP (NP (DT a) (NN sentence)) (SBAR (S (VP (TO to) (VP (VB be) (VP (VBN predicted)))))))) (. !))')
    # tree = Tree.fromstring(tree_string)
    # pp(tree_string)

    # Actual run - parse question and context
    replace, nonreplace = allen_predictor.parse_question_tree_for_phrases(sentence)
    pp(replace)
    pp(nonreplace)
    pp(allen_predictor.context_to_replacement_phrase_sets(context_paragraph))
