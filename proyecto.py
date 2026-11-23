"""Compare a historical baseline with logistic regression on simulated appointments."""
import csv
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UMBRAL = 0.50
FEATURES = ("dias_anticipacion", "inasistencias_previas", "recordatorio")


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def simulate(count, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        days = rng.randint(0, 40)
        previous = rng.choices([0, 1, 2, 3], weights=[65, 23, 9, 3])[0]
        reminder = rng.randint(0, 1)
        probability = sigmoid(-2.0 + 0.045 * days + 0.65 * previous - 0.75 * reminder)
        rows.append((days, previous, reminder, int(rng.random() < probability)))
    rng.shuffle(rows)
    return rows


def prepare(row):
    days, previous, reminder, label = row
    return (1.0, days / 40, previous / 3, float(reminder)), label


def train(rows, epochs=350, learning_rate=0.8):
    weights = [0.0] * 4
    for _ in range(epochs):
        gradients = [0.0] * len(weights)
        for features, label in rows:
            error = predict(weights, features) - label
            for index, value in enumerate(features):
                gradients[index] += error * value
        weights = [
            weight - learning_rate * gradient / len(rows)
            for weight, gradient in zip(weights, gradients)
        ]
    return weights


def predict(weights, features):
    return sigmoid(sum(weight * value for weight, value in zip(weights, features)))


def evaluate(labels, probabilities, threshold=UMBRAL):
    pairs = list(zip(labels, probabilities))
    brier = sum((probability - label) ** 2 for label, probability in pairs) / len(pairs)
    tp = sum(label == 1 and probability >= threshold for label, probability in pairs)
    fp = sum(label == 0 and probability >= threshold for label, probability in pairs)
    fn = sum(label == 1 and probability < threshold for label, probability in pairs)
    tn = len(pairs) - tp - fp - fn
    return {
        "brier": round(brier, 4),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "sensibilidad": round(tp / (tp + fn), 4) if tp + fn else None,
        "matriz_confusion": [[tn, fp], [fn, tp]],
    }


def main():
    rows = simulate(count=1500, seed=42)
    split = int(len(rows) * 0.8)
    training = [prepare(row) for row in rows[:split]]
    test = [prepare(row) for row in rows[split:]]
    prevalence = sum(label for _, label in training) / len(training)
    weights = train(training)
    labels = [label for _, label in test]
    result = {
        "citas_entrenamiento": len(training),
        "citas_prueba": len(test),
        "umbral": UMBRAL,
        "referencia_frecuencia_entrenamiento": evaluate(labels, [prevalence] * len(test)),
        "regresion_logistica": evaluate(labels, [predict(weights, features) for features, _ in test]),
    }
    with (ROOT / "citas_simuladas.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow((*FEATURES, "inasistencia"))
        writer.writerows(rows)
    report = json.dumps(result, ensure_ascii=False, indent=2)
    (ROOT / "resultados.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

