from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from nltk import Tree, sent_tokenize
from pprint import pprint as pp
from typing import Dict, NoReturn, List, Dict, Tuple

# utils
def load_allen_predictor():
    archive = load_archive("./rep_allennlp/elmo-constituency-parser-2018.03.14.tar.gz")
    predictor = Predictor.from_archive(archive, 'constituency-parser')
    return predictor

def get_allen_tree(sentence) -> Tree:
    output = predictor.predict_json(sentence)
    pp(output["trees"])
    return Tree.fromstring(output["trees"])


# parse the tree for a question into the phrasal replacement candidates
# Note that tree is **modified** (all leaves will have an appended "_index"
def _question_to_phrases_for_replacement(start_tree: Tree) -> List[Dict]:
    """
    Parses a tree into the set of phrases we consider for paraphrasing.

    :param tree: The input tree representing a sentence. Parse looking for phrases of certain type
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
            if num_leaves > stop_leaves or label not in tree_label_to_index:            # haven't met stop criterion, so recurse
                _internal_traverse(child)
            else:  # valid stop (few enough leaves, and it's in the replacement list)
                phrase = child.leaves()
                nonlocal phrases_to_paraphrase
                phrases_to_paraphrase += [(phrase, tree_label_to_index[label])]

    tree_with_leaf_indices = _add_indices_to_terminals(start_tree)   # add indices to tree roots
    phrases_to_paraphrase = []                   # accumulation set from tree traversal
    _internal_traverse(tree_with_leaf_indices)   # modifies phrases_to_paraphrase

    output_list = []                                # final accumulation (postprocess)
    for p, label_numeric in phrases_to_paraphrase:  # each p is a tuple (phrase, label_numeric)
        phrase_start_idx = p[0].split("_")[1]
        for i, word in enumerate(p):                # iterate over the words in each phrase, p
            p[i] = word.split("_")[0]
        output_list += [{"phrase": p,
                         "phrase_type": label_numeric,
                         "phrase_start_idx": int(phrase_start_idx)}]

    return output_list


def _question_to_nonreplacement_portions(replacement_candidates, sentence):
    non_replacement_list = []
    last_unused = 0
    for p in replacement_candidates:
        phrase = p["phrase"]
        phrase_len = len(phrase)
        phrase_start = p["phrase_start_idx"]
        if phrase_start > last_unused:
            non_replacement_list += [{"phrase": sentence[last_unused:phrase_start],
                                      "phrase_start_idx": last_unused}]
        last_unused = phrase_start + phrase_len

    return non_replacement_list


def parse_question_tree_for_phrases(tree : Tree) -> Tuple:
    orig_sentence = tree.leaves()       # must do before calling the
    replacement_candidates = _question_to_phrases_for_replacement(tree)
    non_replacement = _question_to_nonreplacement_portions(replacement_candidates, orig_sentence)   # orig sentence must be generated prior to calling replacementcandidates

    return replacement_candidates, non_replacement


def context_to_replacement_phrase_sets(context_sentences : List[str]) -> List[List[str]]:
    """
    :param context_sentences: a list of strings (each string a sentence)
    :return: List (indexed by type of phrase) of List of phrases (type of phrase, int => list of phrases)
    """

    def _internal_traverse(tree):
        for child in tree:
            if not isinstance(child, Tree): # early fail if it's not a tree
                return

            label = child.label()
            num_leaves = len(child.leaves())
            if num_leaves > stop_leaves or label not in tree_label_to_index:            # haven't met stop criterion, so recurse
                _internal_traverse(child)
            else:  # valid stop (few enough leaves, and it's in the replacement list)
                phrase = child.leaves()
                phrase_joined = " ".join(phrase)
                phrase_type_to_phrases[tree_label_to_index[label]][phrase_joined] = phrase  # only save each phrase once


    # Use dictionaries here to avoid multiple copies of the same phrase. Then throw away the key since not needed
    phrase_type_to_phrases = [dict() for i in range(num_replacement_node_types)]

    # Iterate over each sentence tree in the context
    for idx, s in enumerate(context_sentences):
        context_sentences[idx] = get_allen_tree({"sentence": s})
    for t in context_sentences:     # they are trees now
        _internal_traverse(t)

    return [list(x.values()) for x in phrase_type_to_phrases]      # throw away the keys and keep values (the separated phrases)


if __name__ == "__main__":
    # tree and config setup
    stop_leaves = 3
    replacement_node_types = ["NP", "VP", "VBZ", "VBN"] # todo

    # pre-run setup
    num_replacement_node_types = len(replacement_node_types)
    tree_label_to_index = {name: i for i, name in enumerate(replacement_node_types)}

    # allen setup
    predictor = load_allen_predictor()

    # sentence setup
    sentence = {"sentence": "This is a sentence to be predicted from a sentence"}
    tree = get_allen_tree(sentence)
    pp(tree)

    # context setup
    context_paragraph = "This is a context paragraph about coding in python. Python is great. It has really cool typing support."
    context_sentences = sent_tokenize(context_paragraph)

    # example of direct string parse to save compute time on allen predictor
    # tree_string = ('(S (NP (DT This)) (VP (VBZ is) (NP (NP (DT a) (NN sentence)) (SBAR (S (VP (TO to) (VP (VB be) (VP (VBN predicted)))))))) (. !))')
    # tree = Tree.fromstring(tree_string)
    # pp(tree_string)


    # Actual run - parse question and context
    replace, nonreplace = parse_question_tree_for_phrases(tree)

    pp(replace)
    pp(nonreplace)
    pp(context_to_replacement_phrase_sets(context_sentences))

