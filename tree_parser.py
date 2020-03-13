from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from nltk import Tree
from pprint import pprint as pp
from nltk.tree import ParentedTree


def getAllenTree(sentence):
    archive = load_archive("./rep_allennlp/elmo-constituency-parser-2018.03.14.tar.gz")
    predictor = Predictor.from_archive(archive, 'constituency-parser')
    output = predictor.predict_json(sentence)
    tree = Tree.fromstring(output["trees"])

    # for subtree in tree:
    #     if type(subtree) == Tree:
    #         print("Tree:", subtree)

    return tree

def traverseAllenTree(tree):
    np_trees = []
    try:
        tree.label()
    except AttributeError:
        return

    newtree = ParentedTree.convert(tree)
    if newtree.label() == "VP" or newtree.label() == "NP":
        current = newtree
        while current.parent() is not None:
            while current.left_sibling() is not None:
                 if current.left_sibling().label() == "NP" or current.left_sibling().label() == "VP":
                     np_trees.append(current.left_sibling())
                 current = current.left_sibling()
            current = current.parent()

    for child in tree:
        traverseAllenTree(child)
    return np_trees


def extractPhraseFiltered(sentence, label1="NP", label2= "NNP"):

    tree = getAllenTree(sentence)
    for st in tree.subtrees(filter=lambda x: x.label() == label1 and x[0].label == label2):
        print(st)


def find_VP(tree):
    """Recurse on a constituency parse tree until we find verb phrases"""

    # Recursion is annoying because we need to check whether each is a list or not
    def recurse_on_children():
        assert 'children' in tree
        result = []
        for child in tree['children']:
            res = find_VP(child)
            if isinstance(res, tuple):
                result.append(res)
            else:
                result.extend(res)
        return result

    for i in tree.subtrees:
        if i(filter=lambda x: x.node == "VP"):
            return [(tree["word"], "VP")]

    # if 'VP' in tree['attributes']:
    #     # # Now we'll get greedy and see if we can find something better
    #     # if 'children' in tree and len(tree['children']) > 1:
    #     #     recurse_result = _recurse_on_children()
    #     #     if all([x[1] in ('VP', 'NP', 'CC') for x in recurse_result]):
    #     #         return recurse_result
    #     return [(tree['word'], 'VP')]

    # base cases

    if 'NP' in tree['attributes']:
        return [(tree['word'], 'NP')]
    # No children
    if not 'children' in tree:
        return [(tree['word'], tree['attributes'][0])]

    # If a node only has 1 child then we'll have to stick with that
    if len(tree['children']) == 1:
        return recurse_on_children()
    # try recursing on everything
    return recurse_on_children()

def traverse(t):

    newtree = ParentedTree.convert(t)
    try:
        newtree.label()
    except AttributeError:
        return
    else:
        print("height", newtree.height())
        if newtree.height() == 2:   #child nodes
            print("here!!!", newtree)
            return

        else:
            print("no height")

        for child in newtree:
            traverse(child)

def getHeightNode(tree):
    newtree = ParentedTree.convert(tree)
    tree_dict = {}
    for i in newtree.subtrees():
        tree_dict = {}
        print(i.height(), i)
        leaf_index = leaf_values.index('nice')
        tree_location = newtree.leaf_treeposition(leaf_index)

def getNodeLeaves(tree):
    # leafpos = [tree.leaf_treeposition(n) for n, x in enumerate(tree.leaves())]
    # level1_subtrees = [tree[path[:-1]] for path in leafpos]
    # print(leafpos)
    # for x in level1_subtrees:
    #     print(x, end=" ")
    tree_dict = {}
    for i in tree.subtrees():
        if i.label() not in tree_dict:
            tree_dict[i.label()] = [(i.leaves(),len(i.leaves()))]
        else:
            tree_dict[i.label()] += [(i.leaves(),len(i.leaves()))]

    return tree_dict


if __name__ == "__main__":
    sentence = {"sentence": "This is a sentence to be predicted!"}
    tree = getAllenTree(sentence)
    # print(traverseAllenTree(tree))
    # print(find_VP(tree))
    # print(traverse(tree))
    # print(getHeightNode(tree))
    print(getNodeLeaves(tree))
    # print(getNodePosition(tree))