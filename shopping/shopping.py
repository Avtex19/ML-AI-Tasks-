import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


SPLIT_RATIO = 0.4


def run():
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    data, targets = parse_csv(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=SPLIT_RATIO
    )

    model = train_knn_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = assess_model(y_test, predictions)

    correct = sum(predictions == y_test)
    total = len(y_test)

    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"True Positive Rate: {sensitivity * 100:.2f}%")
    print(f"True Negative Rate: {specificity * 100:.2f}%")


def parse_csv(filepath):
    month_mapping = {month: idx - 1 for idx, month in enumerate(calendar.month_abbr) if idx}
    month_mapping["June"] = month_mapping.pop("Jun")

    features = []
    outcomes = []

    with open(filepath, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            features.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                month_mapping[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                int(row["Weekend"] == "TRUE")
            ])
            outcomes.append(int(row["Revenue"] == "TRUE"))

    return features, outcomes


def train_knn_model(features, labels):
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(features, labels)
    return model


def assess_model(actual, predicted):
    true_positive = sum(1 for a, p in zip(actual, predicted) if a == 1 and p == 1)
    true_negative = sum(1 for a, p in zip(actual, predicted) if a == 0 and p == 0)

    total_positive = actual.count(1)
    total_negative = actual.count(0)

    sensitivity = true_positive / total_positive if total_positive else 0
    specificity = true_negative / total_negative if total_negative else 0

    return sensitivity, specificity


if __name__ == "__main__":
    run()
