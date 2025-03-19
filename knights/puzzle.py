from logic import *

# Define symbols for each character being a knight or a knave
AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")
BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")
CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# General constraints: Each character is either a knight or a knave, but not both
base_rules = And(
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),
    Or(CKnight, CKnave),
    Not(And(CKnight, CKnave))
)

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    base_rules,
    Implication(AKnight, And(AKnight, AKnave)),
    Implication(AKnave, Not(And(AKnight, AKnave)))
)

# Puzzle 1
# A says "We are both knaves."
knowledge1 = And(
    base_rules,
    Implication(AKnight, And(AKnave, BKnave)),  # If A were a knight, both must be knaves (impossible)
    Implication(AKnave, Not(And(AKnave, BKnave))),  # If A is a knave, then the statement must be false
    BKnight  # Since A is lying, B must be a knight
)

# Puzzle 2
# A says "We are the same kind."\# B says "We are of different kinds."
knowledge2 = And(
    base_rules,
    Biconditional(AKnight, BKnight),  # A claims they are the same
    Biconditional(BKnight, Not(AKnight))  # B claims they are different
)

# Puzzle 3
# A says something unknown.
# B says "A said 'I am a knave'." and "C is a knave."\# C says "A is a knight."
knowledge3 = And(
    base_rules,
    Implication(BKnight, Biconditional(AKnave, AKnave)),  # If B is a knight, A must have said "I am a knave"
    Implication(BKnight, CKnave),  # If B is a knight, then C must be a knave
    Implication(BKnave, Not(CKnave)),  # If B is a knave, C must be a knight
    Implication(CKnight, AKnight),  # If C is a knight, A must be a knight
    Implication(CKnave, Not(AKnight))  # If C is a knave, A must be a knave
)

# Function to check which statements are true in each knowledge base
def main():
    for puzzle, knowledge in [(0, knowledge0), (1, knowledge1), (2, knowledge2), (3, knowledge3)]:
        print(f"Puzzle {puzzle}")
        for symbol in [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]:
            if model_check(knowledge, symbol):
                print(f"    {symbol}")

if __name__ == "__main__":
    main()