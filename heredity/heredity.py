import csv
import itertools
import sys

PROBABILITIES = {

    # Basic gene probabilities
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of having the trait with two copies of the gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of having the trait with one copy of the gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of having the trait with no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation chance
    "mutation": 0.01
}


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")

    people_data = load_family_data(sys.argv[1])

    # Track gene and trait probabilities for each person
    prob_data = {
        individual: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for individual in people_data
    }

    # Iterate over possible combinations of individuals with the trait
    all_names = set(people_data)
    for trait_set in generate_subsets(all_names):

        # Skip invalid sets based on known trait data
        if any(
                (people_data[person]["trait"] is not None and
                 people_data[person]["trait"] != (person in trait_set))
                for person in all_names
        ):
            continue

        # Iterate over possible gene combinations
        for one_copy in generate_subsets(all_names):
            for two_copies in generate_subsets(all_names - one_copy):
                # Calculate and update joint probability
                joint_prob = calculate_joint_probability(people_data, one_copy, two_copies, trait_set)
                update_probabilities(prob_data, one_copy, two_copies, trait_set, joint_prob)

    # Normalize the probabilities
    normalize_probabilities(prob_data)

    # Output the results
    for individual in people_data:
        print(f"{individual}:")
        for category in prob_data[individual]:
            print(f"  {category.capitalize()}:")
            for value in prob_data[individual][category]:
                probability = prob_data[individual][category][value]
                print(f"    {value}: {probability:.4f}")


def load_family_data(filename):
    """
    Loads gene and trait data from a CSV file.
    Each row contains a name, mother, father, and trait information.
    """
    family_data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            family_data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return family_data


def generate_subsets(s):
    """
    Returns all possible subsets of the given set `s`.
    """
    s = list(s)
    return [
        set(subset) for subset in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def calculate_joint_probability(people_data, one_copy, two_copies, trait_set):
    """
    Computes the joint probability for the given sets of individuals.
    """
    total_prob = 1

    for person in people_data:

        individual_prob = 1
        individual_genes = (2 if person in two_copies else 1 if person in one_copy else 0)
        individual_trait = person in trait_set

        mother = people_data[person]['mother']
        father = people_data[person]['father']

        # Calculate gene probability if the person has no parents
        if not mother and not father:
            individual_prob *= PROBABILITIES['gene'][individual_genes]

        # Otherwise, calculate gene probability based on parents
        else:
            mother_gene_prob = get_inherited_prob(mother, one_copy, two_copies)
            father_gene_prob = get_inherited_prob(father, one_copy, two_copies)

            if individual_genes == 2:
                individual_prob *= mother_gene_prob * father_gene_prob
            elif individual_genes == 1:
                individual_prob *= (1 - mother_gene_prob) * father_gene_prob + (1 - father_gene_prob) * mother_gene_prob
            else:
                individual_prob *= (1 - mother_gene_prob) * (1 - father_gene_prob)

        # Multiply by the trait probability based on the individual's genes
        individual_prob *= PROBABILITIES['trait'][individual_genes][individual_trait]

        total_prob *= individual_prob

    return total_prob


def get_inherited_prob(parent_name, one_copy, two_copies):
    """
    Returns the probability of inheriting a gene from a parent.
    """
    if parent_name in two_copies:
        return 1 - PROBABILITIES['mutation']
    elif parent_name in one_copy:
        return 0.5
    else:
        return PROBABILITIES['mutation']


def update_probabilities(prob_data, one_copy, two_copies, trait_set, joint_prob):
    """
    Updates the probability distributions for each individual.
    """
    for person in prob_data:
        individual_genes = (2 if person in two_copies else 1 if person in one_copy else 0)
        individual_trait = person in trait_set

        prob_data[person]['gene'][individual_genes] += joint_prob
        prob_data[person]['trait'][individual_trait] += joint_prob


def normalize_probabilities(prob_data):
    """
    Normalizes the probability distributions to ensure they sum to 1.
    """
    for person in prob_data:
        gene_prob_total = sum(prob_data[person]['gene'].values())
        trait_prob_total = sum(prob_data[person]['trait'].values())

        prob_data[person]['gene'] = {gene: (prob / gene_prob_total) for gene, prob in prob_data[person]['gene'].items()}
        prob_data[person]['trait'] = {trait: (prob / trait_prob_total) for trait, prob in
                                      prob_data[person]['trait'].items()}


if __name__ == "__main__":
    main()
