import re
import time
import requests
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

CSV_FILE = "diabetes_012_health_indicators_BRFSS2015.csv"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

SAMPLE_SIZE = 10000

SLEEP_BETWEEN_CALLS = 0.2

MAX_RETRIES = 3
REQUEST_TIMEOUT = 300 

def row_to_prompt(row: pd.Series) -> str:
    prompt = """
You are classifying diabetes risk from health indicators.

Class definitions:
- Class 0 = Non-diabetic: generally healthier profile, lower BMI, fewer risk factors
- Class 1 = Prediabetes: intermediate risk, some warning signs, but not as severe as diabetes
- Class 2 = Diabetes: stronger risk profile, usually multiple risk factors and worse overall health

Important rules: 
- Do NOT choose class 1 unless the patient clearly looks intermediate between class 0 and class 2. 
- Class 1 should be the very LAST resort, only if the patient does not fit the other 2 classes. 
- If the patient has many risk factors, choose class 2 over class 1. 
- If the patient has few risk factors and better overall health, choose class 0. 
- If there is a lot of strong evidence of high risk, choose class 2. 
- If there is a lot of strong evidence of low risk, choose class 0. 
- Classify based on the patient features, not just class frequency.

Examples:

Patient health indicators:
HighBP: 0
HighChol: 0
CholCheck: 1
BMI: 22
Smoker: 0
Stroke: 0
HeartDiseaseorAttack: 0
PhysActivity: 1
Fruits: 1
Veggies: 1
HvyAlcoholConsump: 0
AnyHealthcare: 1
NoDocbcCost: 0
GenHlth: 1
MentHlth: 0
PhysHlth: 0
DiffWalk: 0
Sex: 0
Age: 3
Education: 6
Income: 8
Answer: 0

Patient health indicators:
HighBP: 0
HighChol: 0
CholCheck: 1
BMI: 24
Smoker: 0
Stroke: 0
HeartDiseaseorAttack: 0
PhysActivity: 1
Fruits: 1
Veggies: 1
HvyAlcoholConsump: 0
AnyHealthcare: 1
NoDocbcCost: 0
GenHlth: 2
MentHlth: 1
PhysHlth: 1
DiffWalk: 0
Sex: 1
Age: 4
Education: 5
Income: 7
Answer: 0

Patient health indicators:
HighBP: 0
HighChol: 1
CholCheck: 1
BMI: 28
Smoker: 1
Stroke: 0
HeartDiseaseorAttack: 0
PhysActivity: 0
Fruits: 0
Veggies: 1
HvyAlcoholConsump: 0
AnyHealthcare: 1
NoDocbcCost: 0
GenHlth: 3
MentHlth: 4
PhysHlth: 5
DiffWalk: 0
Sex: 1
Age: 8
Education: 4
Income: 5
Answer: 1

Patient health indicators:
HighBP: 1
HighChol: 1
CholCheck: 1
BMI: 30
Smoker: 1
Stroke: 0
HeartDiseaseorAttack: 0
PhysActivity: 0
Fruits: 0
Veggies: 0
HvyAlcoholConsump: 0
AnyHealthcare: 1
NoDocbcCost: 0
GenHlth: 3
MentHlth: 6
PhysHlth: 7
DiffWalk: 0
Sex: 0
Age: 9
Education: 4
Income: 4
Answer: 1

Patient health indicators:
HighBP: 1
HighChol: 1
CholCheck: 1
BMI: 33
Smoker: 1
Stroke: 0
HeartDiseaseorAttack: 1
PhysActivity: 0
Fruits: 0
Veggies: 0
HvyAlcoholConsump: 0
AnyHealthcare: 1
NoDocbcCost: 1
GenHlth: 4
MentHlth: 10
PhysHlth: 15
DiffWalk: 1
Sex: 1
Age: 11
Education: 3
Income: 2
Answer: 2

Patient health indicators:
HighBP: 1
HighChol: 1
CholCheck: 1
BMI: 34
Smoker: 1
Stroke: 1
HeartDiseaseorAttack: 1
PhysActivity: 0
Fruits: 0
Veggies: 0
HvyAlcoholConsump: 0
AnyHealthcare: 1
NoDocbcCost: 1
GenHlth: 5
MentHlth: 12
PhysHlth: 20
DiffWalk: 1
Sex: 1
Age: 12
Education: 2
Income: 1
Answer: 2

Patient health indicators:
HighBP: 1
HighChol: 1
CholCheck: 1
BMI: 32
Smoker: 1
Stroke: 0
HeartDiseaseorAttack: 0
PhysActivity: 0
Fruits: 0
Veggies: 0
HvyAlcoholConsump: 0
AnyHealthcare: 1
NoDocbcCost: 0
GenHlth: 4
MentHlth: 7
PhysHlth: 9
DiffWalk: 1
Sex: 0
Age: 10
Education: 3
Income: 3
Answer: 2

Now classify this patient.

Patient health indicators:
""".strip()

    for col, val in row.items():
        prompt += f"\n{col}: {val}"

    prompt += """

Predict the diabetes risk class:
0 = Non-diabetic
1 = Prediabetes
2 = Diabetes

Return ONLY one number: 0, 1, or 2
""".rstrip()

    return prompt

def extract_label(text: str):
    if text is None:
        return None
    match = re.search(r"\b([012])\b", text.strip())
    return int(match.group(1)) if match else None

def predict_with_llama(prompt: str, max_retries: int = MAX_RETRIES):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()

            data = response.json()
            output_text = data.get("response", "")
            label = extract_label(output_text)

            if label in [0, 1, 2]:
                return label

            print(f"Could not parse valid label from output: {output_text!r}")
            return None

        except requests.exceptions.RequestException as e:
            print(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)

    return None

def main():
    df = pd.read_csv(CSV_FILE)

    num_rows, num_cols = df.shape
    print(f"Total rows: {num_rows}")
    print(f"Total labels (columns): {num_cols}")

    X = df.drop("Diabetes_012", axis=1)
    y = df["Diabetes_012"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print("\nDataset split:")
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    eval_df = X_val.copy()
    eval_df["label"] = y_val.values

    eval_df = X_val.copy()
    eval_df["label"] = y_val.values

    eval_df = eval_df.sample(
        n=min(SAMPLE_SIZE, len(eval_df)),
        random_state=42
    ).reset_index(drop=True)

    print("\nStratified evaluation sample distribution:")
    print(eval_df["label"].value_counts().sort_index())

    y_true = []
    y_pred = []

    for i, row in eval_df.iterrows():
        true_label = int(row["label"])
        features = row.drop("label")

        prompt = row_to_prompt(features)
        pred_label = predict_with_llama(prompt)

        if pred_label is not None:
            y_true.append(true_label)
            y_pred.append(pred_label)
        else:
            print(f"Skipping sample {i} due to invalid response.")

        print(f"Processed {i + 1} / {len(eval_df)} rows")
        time.sleep(SLEEP_BETWEEN_CALLS)

    if len(y_pred) == 0:
        print("\nNo valid predictions were returned.")
        return

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\nResults:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()