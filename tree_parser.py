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

    for subtree in tree:
        if type(subtree) == Tree:
            print("Tree:", subtree)

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



if __name__ == "__main__":
    sentence = {"sentence": "This is a sentence to be predicted!"}
    tree = getAllenTree(sentence)
    print(traverseAllenTree(tree))
