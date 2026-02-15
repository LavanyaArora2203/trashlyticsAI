

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - UPDATE THIS WITH YOUR FILE PATH
# ============================================================================

path="data\\complaints.csv" # Change this to your CSV file path

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================



def generate_label(text):
    text = text.lower()
    
    if any(word in text for word in ["overflow", "overflowing", "full", "spilling"]):
        return "Overflow"
    
    elif any(word in text for word in ["missed", "not collected", "skipped", "delay"]):
        return "Missed pickup"
    
    elif any(word in text for word in ["illegal", "dumped", "unauthorized", "burning"]):
        return "Illegal dumping"
    
    else:
        return "Overflow"  # fallback




try:
    # Load the dataset
    complaints = pd.read_csv(path)
    print(f"✓ Data loaded successfully!")
    print(f"  Total complaints: {len(complaints)}")
    
    # Display first few rows
    print("\nFirst few rows of data:")
    print(complaints.head())

    
    complaints["category"] = complaints["complaint_text"].apply(generate_label)
    
    # Check required columns
    if 'complaint_text' not in complaints.columns or 'category' not in complaints.columns:
        print("\n❌ Error: CSV must have 'complaint_text' and 'category' columns")
        print(f"   Found columns: {complaints.columns.tolist()}")
        exit()
    
except FileNotFoundError:
    print(f"\n❌ Error: File '{path}' not found!")
    print("\nPlease update the CSV_FILE_PATH variable in the script with your file path.")
    print("Example: CSV_FILE_PATH = '/path/to/your/complaints.csv'")
    exit()

# ============================================================================
# STEP 2: EXPLORE DATA IMBALANCE
# ============================================================================

print("\n" + "="*70)
print("[STEP 2] Analyzing class distribution...")
print("="*70)

class_distribution = complaints['category'].value_counts()
print("\nClass Distribution:")
print(class_distribution)
print(f"\nImbalance Ratio: {class_distribution.max() / class_distribution.min():.2f}:1")

# Calculate percentages
print("\nPercentages:")
for category, count in class_distribution.items():
    percentage = (count / len(complaints)) * 100
    print(f"  {category}: {percentage:.1f}%")

# ============================================================================
# STEP 3: PREPARE DATA
# ============================================================================

print("\n" + "="*70)
print("[STEP 3] Preparing data for training...")
print("="*70)

# Extract features and labels
X = complaints['complaint_text']
y = complaints['category']

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"\n✓ Data split complete:")
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")

print(f"\nTraining set distribution:")
print(pd.Series(y_train).value_counts())

# ============================================================================
# STEP 4: TEXT VECTORIZATION
# ============================================================================

print("\n" + "="*70)
print("[STEP 4] Converting text to numerical features (TF-IDF)...")
print("="*70)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=1000,      # Use top 1000 words
    ngram_range=(1, 2),     # Use unigrams and bigrams
    min_df=1,               # Minimum document frequency
    lowercase=True,         # Convert to lowercase
    stop_words='english'    # Remove English stop words
)

# Fit and transform training data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"\n✓ Vectorization complete:")
print(f"  Feature dimensions: {X_train_vec.shape[1]}")
print(f"  Training matrix shape: {X_train_vec.shape}")
print(f"  Testing matrix shape: {X_test_vec.shape}")

# ============================================================================
# STEP 5: HANDLE IMBALANCED DATA WITH SMOTE
# ============================================================================

print("\n" + "="*70)
print("[STEP 5] Handling imbalanced data using SMOTE...")
print("="*70)

print("\nBefore SMOTE:")
from collections import Counter
print(f"  {Counter(y_train)}")

# Calculate k_neighbors based on smallest class
min_samples = min(Counter(y_train).values())
k_neighbors = min(5, min_samples - 1)

if k_neighbors < 1:
    print("\n⚠ Warning: Too few samples for SMOTE. Using class weights instead.")
    use_smote = False
else:
    use_smote = True
    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)
    
    print("\nAfter SMOTE:")
    print(f"  {Counter(y_train_balanced)}")
    print(f"\n✓ Dataset balanced successfully!")
    print(f"  New training size: {X_train_balanced.shape[0]}")

# ============================================================================
# STEP 6: TRAIN THE CLASSIFIER
# ============================================================================

print("\n" + "="*70)
print("[STEP 6] Training the complaint classifier...")
print("="*70)

if use_smote:
    # Train with balanced data
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train_balanced, y_train_balanced)
else:
    # Train with class weights if SMOTE not possible
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced'
    )
    model.fit(X_train_vec, y_train)

print("\n✓ Model training complete!")

# ============================================================================
# STEP 7: EVALUATE THE MODEL
# ============================================================================

print("\n" + "="*70)
print("[STEP 7] Evaluating model performance...")
print("="*70)

# Make predictions
y_pred = model.predict(X_test_vec)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\n📊 Overall Metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1-Score (Macro): {f1_macro:.4f}")
print(f"  F1-Score (Weighted): {f1_weighted:.4f}")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
categories = sorted(y_test.unique())

# Print confusion matrix with labels
print("\n" + " "*20 + "Predicted")
print(" "*15 + "  ".join(f"{cat[:10]:>10}" for cat in categories))
for i, category in enumerate(categories):
    print(f"{category[:10]:>12}  " + "  ".join(f"{cm[i][j]:>10}" for j in range(len(categories))))

# ============================================================================
# STEP 8: SAVE THE MODEL
# ============================================================================

print("\n" + "="*70)
print("[STEP 8] Saving the trained model...")
print("="*70)

# Save model and vectorizer
joblib.dump(model, 'complaint_classifier_model.pkl')
joblib.dump(vectorizer, 'complaint_vectorizer.pkl')

print("\n✓ Model saved successfully!")
print("  Files created:")
print("    - complaint_classifier_model.pkl")
print("    - complaint_vectorizer.pkl")

# ============================================================================
# STEP 9: TEST WITH SAMPLE PREDICTIONS
# ============================================================================

print("\n" + "="*70)
print("[STEP 9] Testing with sample complaints...")
print("="*70)

# Sample complaints for testing
sample_complaints = [
    "Garbage bins are overflowing at the park",
    "Waste collection truck missed our street yesterday",
    "Someone dumped construction debris near the playground",
    "Trash cans are full and need to be emptied",
    "Pickup was skipped last Monday",
    "Illegal waste dumping on the corner"
]

# Make predictions
sample_vec = vectorizer.transform(sample_complaints)
sample_predictions = model.predict(sample_vec)
sample_probabilities = model.predict_proba(sample_vec)

print("\n🧪 Sample Predictions:")
for i, (complaint, prediction) in enumerate(zip(sample_complaints, sample_predictions)):
    probs = sample_probabilities[i]
    max_prob = max(probs)
    print(f"\n{i+1}. '{complaint}'")
    print(f"   → Predicted: {prediction} (confidence: {max_prob:.2%})")

# ============================================================================
# STEP 10: USAGE INSTRUCTIONS
# ============================================================================

print("\n" + "="*70)
print("✅ CLASSIFIER READY TO USE!")
print("="*70)

print("""
To use this classifier in your code:

```python
import joblib

# Load the saved model
model = joblib.load('complaint_classifier_model.pkl')
vectorizer = joblib.load('complaint_vectorizer.pkl')

# Classify new complaints
new_complaints = [
    "Garbage truck didn't come today",
    "Bins overflowing with trash"
]

# Transform and predict
new_vec = vectorizer.transform(new_complaints)
predictions = model.predict(new_vec)
probabilities = model.predict_proba(new_vec)

# Display results
for complaint, pred, prob in zip(new_complaints, predictions, probabilities):
    confidence = max(prob)
    print(f"Complaint: {complaint}")
    print(f"Category: {pred} ({confidence:.2%} confidence)")
```

Classes the model can predict:
- Overflow
- Missed pickup
- Illegal dumping
""")

import pickle

with open("complaint_model.pkl", "rb") as f:
    model = pickle.load(f)

# Now re-save cleanly
with open("complaint_model_new.pkl", "wb") as f:
    pickle.dump(model, f)


print("\n" + "="*70)
print("DONE! 🎉")
print("="*70)
