import numpy as np

def run_train_test(training_input, testing_input):

    def parse_input(data):
        header = data[0]
        num_classes = int(header[0])
        counts = [int(x) for x in header[1:]]

        features, labels = [], []
        row = 1

        for class_idx in range(num_classes):
            for _ in range(counts[class_idx]):
                features.append(data[row])
                labels.append(class_idx)
                row += 1

        return np.array(features), np.array(labels)

    X_train, y_train = parse_input(training_input)
    X_test, y_test = parse_input(testing_input)

    centroids = [
        np.mean(X_train[y_train == i], axis=0)
        for i in range(3)
    ]

    def compare(x, c1, c2):
        midpoint = (centroids[c1] + centroids[c2]) / 2.0
        w = centroids[c2] - centroids[c1]

        return c2 if np.dot(x - midpoint, w) > 0 else c1

    predictions = []

    for x in X_test:
        winner_ab = compare(x, 0, 1)
        final_choice = (
            compare(x, 0, 2)
            if winner_ab == 0
            else compare(x, 1, 2)
        )
        predictions.append(final_choice)

    y_pred = np.array(predictions)

    tp = np.sum((y_test == 0) & (y_pred == 0))
    tn = np.sum((y_test != 0) & (y_pred != 0))
    fp = np.sum((y_test != 0) & (y_pred == 0))
    fn = np.sum((y_test == 0) & (y_pred != 0))

    accuracy = np.mean(y_pred == y_test)

    return {
        "tpr": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "error_rate": float(1 - accuracy),
        "accuracy": float(accuracy),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
    }


if __name__ == "__main__":
    import os

    data_dir = "./data/"

    def load_raw(fname):
        with open(os.path.join(data_dir, fname), "r") as f:
            d = [[float(y) for y in x.strip().split()] for x in f]
            d[0] = [int(x) for x in d[0]]
            return d
