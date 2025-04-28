import nltk
import sys
import re

# Terminal grammar rules
TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

# Nonterminal grammar rules
NONTERMINALS = """
S -> NP VP | NP VP Conj NP VP | NP VP Conj VP
NP -> N | Det N | Det AP N | P NP | NP P NP
VP -> V | Adv VP | V Adv | VP NP | V NP Adv
AP -> Adj | AP Adj
"""

# Build the parser
grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)

def main():
    # Check if a file is given
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as file:
            sentence = file.read()
    else:
        sentence = input("Sentence: ")

    # Process the input sentence
    words = preprocess(sentence)

    # Try parsing the sentence
    try:
        trees = list(parser.parse(words))
    except ValueError as e:
        print(e)
        return

    if not trees:
        print("Could not parse sentence.")
        return

    # Show trees and noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))

def preprocess(sentence):
    """
    Lowercase the sentence and tokenize it,
    keeping only words that have alphabetic letters.
    """
    pattern = re.compile("[a-zA-Z]")
    tokens = nltk.word_tokenize(sentence)
    return [word.lower() for word in tokens if pattern.search(word)]

def np_chunk(tree):
    """
    Find and return noun phrase chunks in the parse tree.
    Chunks are NP subtrees without nested NPs.
    """
    chunks = []
    # Use ParentedTree to move upward
    ptree = nltk.tree.ParentedTree.convert(tree)
    for subtree in ptree.subtrees():
        if subtree.label() == "N":
            chunks.append(subtree.parent())
    return chunks

if __name__ == "__main__":
    main()
