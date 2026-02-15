"""
COMPLETE GARBAGE COMPLAINT PRIORITY SYSTEM
Integrates: Text Classification + Image Classification + Priority Scoring

This script shows how to use all components together:
1. Classify complaint text
2. Analyze garbage image
3. Calculate priority score
4. Return HIGH/MEDIUM/LOW priority
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ============================================================================
# PRIORITY SCORER (Standalone - No External Dependencies)
# ============================================================================

class PriorityScorer:
    """
    Calculate priority scores for garbage complaints
    
    Formula:
    priority_score = 0.3 * sentiment + 0.3 * severity + 0.2 * density + 0.2 * overflow
    """
    
    def __init__(self):
        self.weights = {
            'sentiment': 0.3,
            'severity': 0.3,
            'density': 0.2,
            'overflow': 0.2
        }
    
    def calculate_priority(self, sentiment=0.5, severity=0.5, density=0.5, overflow=0.5):
        """
        Calculate priority score
        
        Args:
            sentiment: 0-1 (higher = more negative/urgent)
            severity: 0-1 (from image analysis)
            density: 0-1 (normalized population density)
            overflow: 0-1 (probability of overflow)
        
        Returns:
            dict with score and level (HIGH/MEDIUM/LOW)
        """
        # Ensure all inputs are 0-1
        sentiment = np.clip(sentiment, 0, 1)
        severity = np.clip(severity, 0, 1)
        density = np.clip(density, 0, 1)
        overflow = np.clip(overflow, 0, 1)
        
        # Calculate weighted score
        score = (
            self.weights['sentiment'] * sentiment +
            self.weights['severity'] * severity +
            self.weights['density'] * density +
            self.weights['overflow'] * overflow
        )
        
        # Determine level
        if score >= 0.7:
            level = 'HIGH'
        elif score >= 0.4:
            level = 'MEDIUM'
        else:
            level = 'LOW'
        
        return {
            'priority_score': float(score),
            'priority_level': level,
            'breakdown': {
                'sentiment': sentiment,
                'severity': severity,
                'density': density,
                'overflow': overflow
            }
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_population_density(density, min_density=5000, max_density=30000):
    """
    Normalize population density to 0-1 range
    
    Args:
        density: Raw population density value
        min_density: Minimum expected density (default: 5000)
        max_density: Maximum expected density (default: 30000)
    
    Returns:
        Normalized value between 0 and 1
    """
    if max_density == min_density:
        return 0.5
    normalized = (density - min_density) / (max_density - min_density)
    return np.clip(normalized, 0.0, 1.0)

def sentiment_from_text_negative(text):
    """
    Simple sentiment analysis (dummy for demo)
    Returns 0-1 where higher = more negative/urgent
    
    In production: Use your NLP model's confidence or a sentiment API
    """
    # Simple heuristic based on urgent words
    urgent_words = ['overflow', 'urgent', 'emergency', 'severe', 'immediately', 
                    'dangerous', 'smell', 'hazard', 'rats', 'everywhere']
    
    text_lower = text.lower()
    urgency_count = sum(1 for word in urgent_words if word in text_lower)
    
    # More urgent words = higher score
    score = min(0.3 + urgency_count * 0.15, 0.95)
    
    return score

def category_to_overflow_probability(category):
    """
    Convert complaint category to overflow probability
    
    Args:
        category: 'Overflow', 'Missed pickup', or 'Illegal dumping'
    
    Returns:
        Probability between 0 and 1
    """
    mapping = {
        'Overflow': 1.0,
        'overflow': 1.0,
        'Missed pickup': 0.3,
        'missed pickup': 0.3,
        'Illegal dumping': 0.1,
        'illegal dumping': 0.1
    }
    
    return mapping.get(category, 0.5)

# ============================================================================
# COMPLETE INTEGRATION EXAMPLE
# ============================================================================

def process_single_complaint(
    complaint_text,
    image_severity,
    population_density,
    category,
    sentiment_score=None
):
    """
    Process a single complaint through the complete pipeline
    
    Args:
        complaint_text: The complaint description
        image_severity: Severity from image classifier (0-1)
        population_density: Raw population density value
        category: Complaint category from text classifier
        sentiment_score: Optional pre-calculated sentiment (0-1)
    
    Returns:
        Complete result with priority
    """
    # Initialize scorer
    scorer = PriorityScorer()
    
    # Calculate sentiment if not provided
    if sentiment_score is None:
        sentiment_score = sentiment_from_text_negative(complaint_text)
    
    # Normalize population density
    density_normalized = normalize_population_density(population_density)
    
    # Get overflow probability from category
    overflow_prob = category_to_overflow_probability(category)
    
    # Calculate priority
    priority = scorer.calculate_priority(
        sentiment=sentiment_score,
        severity=image_severity,
        density=density_normalized,
        overflow=overflow_prob
    )
    
    # Combine all results
    result = {
        'complaint_text': complaint_text,
        'category': category,
        'image_severity': image_severity,
        'population_density': population_density,
        'sentiment_score': sentiment_score,
        'overflow_probability': overflow_prob,
        'priority_score': priority['priority_score'],
        'priority_level': priority['priority_level'],
        'component_breakdown': priority['breakdown']
    }
    
    return result

def process_batch_complaints(complaints_df):
    """
    Process multiple complaints from a DataFrame
    
    Expected columns:
    - complaint_text
    - category (from text classifier)
    - image_severity (from image classifier)
    - population_density
    - sentiment_score (optional)
    
    Returns:
        DataFrame with priority scores
    """
    scorer = PriorityScorer()
    results = []
    
    for idx, row in complaints_df.iterrows():
        # Get sentiment
        sentiment = row.get('sentiment_score')
        if sentiment is None or pd.isna(sentiment):
            sentiment = sentiment_from_text_negative(row['complaint_text'])
        
        # Normalize density
        density_norm = normalize_population_density(row['population_density'])
        
        # Get overflow probability
        overflow = category_to_overflow_probability(row['category'])
        
        # Calculate priority
        priority = scorer.calculate_priority(
            sentiment=sentiment,
            severity=row['image_severity'],
            density=density_norm,
            overflow=overflow
        )
        
        # Store result
        result = row.to_dict()
        result['sentiment_score'] = sentiment
        result['overflow_probability'] = overflow
        result['density_normalized'] = density_norm
        result['priority_score'] = priority['priority_score']
        result['priority_level'] = priority['priority_level']
        
        results.append(result)
    
    return pd.DataFrame(results)

# ============================================================================
# DEMO
# ============================================================================

def create_demo_data():
    """
    Create sample data for demonstration
    """
    np.random.seed(42)
    
    complaints = [
        {
            'complaint_id': 'C001',
            'complaint_text': 'Garbage overflowing near market area, urgent attention needed',
            'category': 'Overflow',
            'image_severity': 0.85,
            'population_density': 22000,
            'location': 'Dwarka Sector 3'
        },
        {
            'complaint_id': 'C002',
            'complaint_text': 'Missed pickup since 3 days',
            'category': 'Missed pickup',
            'image_severity': 0.45,
            'population_density': 15000,
            'location': 'Rohini Sector 7'
        },
        {
            'complaint_id': 'C003',
            'complaint_text': 'Someone dumped construction waste illegally',
            'category': 'Illegal dumping',
            'image_severity': 0.65,
            'population_density': 8000,
            'location': 'Vasant Vihar'
        },
        {
            'complaint_id': 'C004',
            'complaint_text': 'Emergency! Bins overflowing everywhere, hazardous smell',
            'category': 'Overflow',
            'image_severity': 0.92,
            'population_density': 28000,
            'location': 'Connaught Place'
        },
        {
            'complaint_id': 'C005',
            'complaint_text': 'Pickup was delayed by one day',
            'category': 'Missed pickup',
            'image_severity': 0.35,
            'population_density': 12000,
            'location': 'Karol Bagh'
        },
    ]
    
    return pd.DataFrame(complaints)

def display_priority_results(df):
    """
    Display results in a nice format
    """
    print("\n" + "="*100)
    print("PRIORITY SCORING RESULTS")
    print("="*100)
    
    # Sort by priority score
    df_sorted = df.sort_values('priority_score', ascending=False)
    
    print(f"\n{'ID':<6} {'Location':<20} {'Category':<15} {'Priority':<8} {'Score':<6} {'Details'}")
    print("-" * 100)
    
    for idx, row in df_sorted.iterrows():
        details = f"S:{row['image_severity']:.2f} D:{row.get('density_normalized', 0):.2f} O:{row['overflow_probability']:.2f}"
        print(f"{row['complaint_id']:<6} {row['location']:<20} {row['category']:<15} "
              f"{row['priority_level']:<8} {row['priority_score']:.3f}  {details}")
    
    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    high = len(df_sorted[df_sorted['priority_level'] == 'HIGH'])
    medium = len(df_sorted[df_sorted['priority_level'] == 'MEDIUM'])
    low = len(df_sorted[df_sorted['priority_level'] == 'LOW'])
    total = len(df_sorted)
    
    print(f"\nTotal Complaints: {total}")
    print(f"  HIGH:   {high} ({high/total*100:.1f}%)")
    print(f"  MEDIUM: {medium} ({medium/total*100:.1f}%)")
    print(f"  LOW:    {low} ({low/total*100:.1f}%)")
    print(f"\nAverage Priority Score: {df_sorted['priority_score'].mean():.3f}")

def main():
    """
    Main demonstration
    """
    print("="*100)
    print("COMPLETE GARBAGE COMPLAINT PRIORITY SYSTEM")
    print("="*100)
    
    print("\n[1] Creating demo data...")
    df = create_demo_data()
    print(f"✓ Created {len(df)} sample complaints")
    
    print("\n[2] Processing complaints...")
    df_prioritized = process_batch_complaints(df)
    print("✓ All complaints processed")
    
    print("\n[3] Displaying results...")
    display_priority_results(df_prioritized)
    
    # Example: Single complaint
    print("\n" + "="*100)
    print("EXAMPLE: PROCESSING SINGLE COMPLAINT")
    print("="*100)
    
    result = process_single_complaint(
        complaint_text="Emergency! Garbage bins overflowing, rats everywhere!",
        image_severity=0.89,
        population_density=25000,
        category="Overflow"
    )
    
    print(f"\nComplaint: {result['complaint_text']}")
    print(f"Category: {result['category']}")
    print(f"Image Severity: {result['image_severity']:.3f}")
    print(f"Population Density: {result['population_density']}")
    print(f"\n✅ PRIORITY: {result['priority_level']}")
    print(f"   Score: {result['priority_score']:.3f}")
    print(f"\nBreakdown:")
    print(f"  Sentiment (30%):   {result['component_breakdown']['sentiment']:.3f}")
    print(f"  Severity (30%):    {result['component_breakdown']['severity']:.3f}")
    print(f"  Density (20%):     {result['component_breakdown']['density']:.3f}")
    print(f"  Overflow (20%):    {result['component_breakdown']['overflow']:.3f}")
    
    # Save results
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)
    
    df_prioritized.to_csv('prioritized_complaints.csv', index=False)
    print("✓ Saved to: prioritized_complaints.csv")
    
    # Usage guide
    print("\n" + "="*100)
    print("HOW TO USE IN YOUR PROJECT")
    print("="*100)
    
    print("""
# Step 1: Load your models
import joblib
import pickle

complaint_classifier = joblib.load('complaint_classifier_model.pkl')
complaint_vectorizer = joblib.load('complaint_vectorizer.pkl')

with open('garbage_classifier.pkl', 'rb') as f:
    image_classifier = pickle.load(f)

# Step 2: Process a new complaint
complaint_text = "Bins overflowing for 3 days"
image_path = "complaint_photo.jpg"
population_density = 18000

# Classify text
text_vec = complaint_vectorizer.transform([complaint_text])
category = complaint_classifier.predict(text_vec)[0]

# Classify image
image_result = image_classifier.predict(image_path)
image_severity = image_result['severity_score']

# Calculate priority
result = process_single_complaint(
    complaint_text=complaint_text,
    image_severity=image_severity,
    population_density=population_density,
    category=category
)

print(f"Priority: {result['priority_level']}")
print(f"Score: {result['priority_score']:.3f}")

# Step 3: Process batch from CSV
df = pd.read_csv('new_complaints.csv')
df_prioritized = process_batch_complaints(df)
df_prioritized.to_csv('output.csv', index=False)
    """)
    
    print("\n" + "="*100)
    print("✅ SYSTEM READY!")
    print("="*100)

if __name__ == "__main__":
    main()