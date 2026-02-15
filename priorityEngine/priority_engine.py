"""
PRIORITY SCORING ENGINE
Calculates priority scores for garbage complaints

Formula:
priority_score = 0.3 * sentiment_score
               + 0.3 * image_severity
               + 0.2 * population_density
               + 0.2 * overflow_probability

Returns: HIGH / MEDIUM / LOW priority
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

# ============================================================================
# PRIORITY SCORING ENGINE
# ============================================================================

class PriorityScorer:
    """
    Calculates priority scores for complaints based on multiple factors
    """
    
    def __init__(self, weights=None):
        """
        Initialize with custom weights or use defaults
        
        Args:
            weights: dict with keys: sentiment, severity, density, overflow
        """
        if weights is None:
            self.weights = {
                'sentiment': 0.3,
                'severity': 0.3,
                'density': 0.2,
                'overflow': 0.2
            }
        else:
            self.weights = weights
        
        # Priority thresholds
        self.thresholds = {
            'high': 0.7,     # >= 0.7 is HIGH
            'medium': 0.4    # >= 0.4 is MEDIUM, < 0.4 is LOW
        }
    
    def normalize_value(self, value, min_val=0, max_val=1):
        """
        Normalize a value to 0-1 range
        
        Args:
            value: The value to normalize
            min_val: Minimum value in the range
            max_val: Maximum value in the range
        
        Returns:
            Normalized value between 0 and 1
        """
        if max_val == min_val:
            return 0.5  # If all values are the same, return middle
        
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)
    
    def normalize_sentiment(self, sentiment_score):
        """
        Normalize sentiment score to 0-1
        
        Assumes sentiment_score is between -1 (very negative) and 1 (very positive)
        For complaints, negative sentiment = higher priority
        """
        # Convert -1 to 1 range to 0 to 1 range
        # -1 (very negative) -> 1 (high priority)
        # 1 (very positive) -> 0 (low priority)
        normalized = (1 - sentiment_score) / 2
        return np.clip(normalized, 0.0, 1.0)
    
    def calculate_priority_score(self, 
                                 sentiment_score=None,
                                 image_severity=None,
                                 population_density=None,
                                 overflow_probability=None):
        """
        Calculate priority score from individual components
        
        Args:
            sentiment_score: Sentiment from text (-1 to 1, or 0 to 1 normalized)
            image_severity: Severity from image (0 to 1)
            population_density: Population density (will be normalized)
            overflow_probability: Probability of overflow category (0 to 1)
        
        Returns:
            dict with score and priority level
        """
        components = {}
        
        # Sentiment (normalize if needed)
        if sentiment_score is not None:
            if -1 <= sentiment_score <= 1:
                # Sentiment is in -1 to 1 range, normalize it
                components['sentiment'] = self.normalize_sentiment(sentiment_score)
            else:
                # Assume already normalized to 0-1
                components['sentiment'] = np.clip(sentiment_score, 0.0, 1.0)
        else:
            components['sentiment'] = 0.5  # Neutral default
        
        # Image severity (already 0-1)
        if image_severity is not None:
            components['severity'] = np.clip(image_severity, 0.0, 1.0)
        else:
            components['severity'] = 0.5  # Medium default
        
        # Population density (will be normalized in batch processing)
        if population_density is not None:
            components['density'] = np.clip(population_density, 0.0, 1.0)
        else:
            components['density'] = 0.5  # Medium default
        
        # Overflow probability (already 0-1)
        if overflow_probability is not None:
            components['overflow'] = np.clip(overflow_probability, 0.0, 1.0)
        else:
            components['overflow'] = 0.5  # Medium default
        
        # Calculate weighted score
        priority_score = (
            self.weights['sentiment'] * components['sentiment'] +
            self.weights['severity'] * components['severity'] +
            self.weights['density'] * components['density'] +
            self.weights['overflow'] * components['overflow']
        )
        
        # Determine priority level
        if priority_score >= self.thresholds['high']:
            priority_level = 'HIGH'
        elif priority_score >= self.thresholds['medium']:
            priority_level = 'MEDIUM'
        else:
            priority_level = 'LOW'
        
        return {
            'priority_score': float(priority_score),
            'priority_level': priority_level,
            'components': components,
            'weights': self.weights
        }
    
    def process_batch(self, complaints_df):
        """
        Process a batch of complaints from a DataFrame
        
        Expected columns:
        - sentiment_score (optional)
        - image_severity (optional)
        - population_density (optional)
        - overflow_probability or category (optional)
        
        Returns:
            DataFrame with added priority columns
        """
        df = complaints_df.copy()
        
        # Normalize population density across all complaints
        if 'population_density' in df.columns:
            min_density = df['population_density'].min()
            max_density = df['population_density'].max()
            df['population_density_normalized'] = df['population_density'].apply(
                lambda x: self.normalize_value(x, min_density, max_density)
            )
        else:
            df['population_density_normalized'] = 0.5
        
        # Calculate overflow probability if we have category
        if 'category' in df.columns and 'overflow_probability' not in df.columns:
            # Convert category to overflow probability
            df['overflow_probability'] = df['category'].apply(
                lambda x: 1.0 if x == 'Overflow' else 
                         0.3 if x == 'Missed pickup' else 0.1
            )
        elif 'overflow_probability' not in df.columns:
            df['overflow_probability'] = 0.5
        
        # Normalize sentiment if needed
        if 'sentiment_score' in df.columns:
            # Check if sentiment is in -1 to 1 range
            if df['sentiment_score'].min() < 0:
                df['sentiment_normalized'] = df['sentiment_score'].apply(
                    self.normalize_sentiment
                )
            else:
                df['sentiment_normalized'] = df['sentiment_score']
        else:
            df['sentiment_normalized'] = 0.5
        
        # Use image_severity as is (already 0-1) or default
        if 'image_severity' not in df.columns:
            df['image_severity'] = 0.5
        
        # Calculate priority for each row
        priorities = []
        
        for idx, row in df.iterrows():
            result = self.calculate_priority_score(
                sentiment_score=row.get('sentiment_normalized', 0.5),
                image_severity=row.get('image_severity', 0.5),
                population_density=row.get('population_density_normalized', 0.5),
                overflow_probability=row.get('overflow_probability', 0.5)
            )
            priorities.append(result)
        
        # Add results to dataframe
        df['priority_score'] = [p['priority_score'] for p in priorities]
        df['priority_level'] = [p['priority_level'] for p in priorities]
        
        # Add component scores for transparency
        df['sentiment_component'] = [p['components']['sentiment'] for p in priorities]
        df['severity_component'] = [p['components']['severity'] for p in priorities]
        df['density_component'] = [p['components']['density'] for p in priorities]
        df['overflow_component'] = [p['components']['overflow'] for p in priorities]
        
        return df
    
    def get_statistics(self, complaints_df):
        """
        Get statistics on priority distribution
        """
        stats = {
            'total_complaints': len(complaints_df),
            'high_priority': len(complaints_df[complaints_df['priority_level'] == 'HIGH']),
            'medium_priority': len(complaints_df[complaints_df['priority_level'] == 'MEDIUM']),
            'low_priority': len(complaints_df[complaints_df['priority_level'] == 'LOW']),
            'average_score': complaints_df['priority_score'].mean(),
            'min_score': complaints_df['priority_score'].min(),
            'max_score': complaints_df['priority_score'].max(),
        }
        
        # Add percentages
        stats['high_priority_pct'] = (stats['high_priority'] / stats['total_complaints'] * 100) if stats['total_complaints'] > 0 else 0
        stats['medium_priority_pct'] = (stats['medium_priority'] / stats['total_complaints'] * 100) if stats['total_complaints'] > 0 else 0
        stats['low_priority_pct'] = (stats['low_priority'] / stats['total_complaints'] * 100) if stats['total_complaints'] > 0 else 0
        
        return stats
    
    def save(self, filepath='priority_scorer.json'):
        """Save scorer configuration"""
        config = {
            'weights': self.weights,
            'thresholds': self.thresholds
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def load(filepath='priority_scorer.json'):
        """Load scorer configuration"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        scorer = PriorityScorer(weights=config['weights'])
        scorer.thresholds = config['thresholds']
        return scorer

# ============================================================================
# DEMO AND TESTING
# ============================================================================

def create_sample_data(n=20):
    """
    Create sample complaint data for demonstration
    """
    np.random.seed(42)
    
    locations = ['Dwarka Sector 3', 'Rohini Sector 7', 'Vasant Vihar', 
                 'Connaught Place', 'Karol Bagh']
    categories = ['Overflow', 'Missed pickup', 'Illegal dumping']
    
    data = {
        'complaint_id': [f'C{i:04d}' for i in range(1, n+1)],
        'complaint_text': [
            f"Garbage issue in area {i}" for i in range(n)
        ],
        'location': np.random.choice(locations, n),
        'category': np.random.choice(categories, n, p=[0.6, 0.25, 0.15]),
        'sentiment_score': np.random.uniform(-0.8, 0.2, n),  # Complaints are usually negative
        'image_severity': np.random.uniform(0.3, 0.95, n),
        'population_density': np.random.randint(5000, 30000, n),
        'timestamp': pd.date_range(start='2025-01-01', periods=n, freq='6H')
    }
    
    return pd.DataFrame(data)

def display_results(df):
    """
    Display priority scoring results
    """
    print("\n" + "="*90)
    print("PRIORITY SCORING RESULTS")
    print("="*90)
    
    # Sort by priority score (highest first)
    df_sorted = df.sort_values('priority_score', ascending=False)
    
    # Display top results
    print(f"\n{'ID':<8} {'Category':<15} {'Location':<20} {'Priority':<8} {'Score':<8}")
    print("-" * 90)
    
    for idx, row in df_sorted.head(10).iterrows():
        print(f"{row['complaint_id']:<8} {row['category']:<15} {row['location']:<20} "
              f"{row['priority_level']:<8} {row['priority_score']:.3f}")
    
    if len(df) > 10:
        print(f"\n... and {len(df) - 10} more complaints")

def display_statistics(stats):
    """
    Display priority statistics
    """
    print("\n" + "="*90)
    print("PRIORITY STATISTICS")
    print("="*90)
    
    print(f"\nTotal Complaints: {stats['total_complaints']}")
    print(f"\nPriority Distribution:")
    print(f"  HIGH:   {stats['high_priority']:>4} ({stats['high_priority_pct']:>5.1f}%)")
    print(f"  MEDIUM: {stats['medium_priority']:>4} ({stats['medium_priority_pct']:>5.1f}%)")
    print(f"  LOW:    {stats['low_priority']:>4} ({stats['low_priority_pct']:>5.1f}%)")
    
    print(f"\nPriority Scores:")
    print(f"  Average: {stats['average_score']:.3f}")
    print(f"  Minimum: {stats['min_score']:.3f}")
    print(f"  Maximum: {stats['max_score']:.3f}")

def display_component_analysis(df):
    """
    Display analysis of priority components
    """
    print("\n" + "="*90)
    print("COMPONENT ANALYSIS")
    print("="*90)
    
    print(f"\nAverage Component Scores:")
    print(f"  Sentiment:          {df['sentiment_component'].mean():.3f}")
    print(f"  Image Severity:     {df['severity_component'].mean():.3f}")
    print(f"  Population Density: {df['density_component'].mean():.3f}")
    print(f"  Overflow Prob:      {df['overflow_component'].mean():.3f}")
    
    print(f"\nComponent Contribution to High Priority Complaints:")
    high_priority = df[df['priority_level'] == 'HIGH']
    
    if len(high_priority) > 0:
        print(f"  Sentiment:          {high_priority['sentiment_component'].mean():.3f}")
        print(f"  Image Severity:     {high_priority['severity_component'].mean():.3f}")
        print(f"  Population Density: {high_priority['density_component'].mean():.3f}")
        print(f"  Overflow Prob:      {high_priority['overflow_component'].mean():.3f}")

def main():
    """
    Main demonstration
    """
    print("="*90)
    print("PRIORITY SCORING ENGINE DEMO")
    print("="*90)
    
    # Create sample data
    print("\n[1] Creating sample complaint data...")
    df = create_sample_data(n=50)
    print(f"✓ Created {len(df)} sample complaints")
    
    # Initialize priority scorer
    print("\n[2] Initializing Priority Scorer...")
    scorer = PriorityScorer()
    print("✓ Priority Scorer initialized with weights:")
    print(f"   Sentiment:          {scorer.weights['sentiment']}")
    print(f"   Image Severity:     {scorer.weights['severity']}")
    print(f"   Population Density: {scorer.weights['density']}")
    print(f"   Overflow Prob:      {scorer.weights['overflow']}")
    
    # Process complaints
    print("\n[3] Calculating priority scores...")
    df_prioritized = scorer.process_batch(df)
    print("✓ Priority scores calculated")
    
    # Display results
    display_results(df_prioritized)
    
    # Get and display statistics
    stats = scorer.get_statistics(df_prioritized)
    display_statistics(stats)
    
    # Component analysis
    display_component_analysis(df_prioritized)
    
    # Save results
    print("\n" + "="*90)
    print("SAVING RESULTS")
    print("="*90)
    
    output_file = 'prioritized_complaints.csv'
    df_prioritized.to_csv(output_file, index=False)
    print(f"✓ Saved to: {output_file}")
    
    # Save scorer config
    scorer.save('priority_scorer_config.json')
    print(f"✓ Saved scorer config to: priority_scorer_config.json")
    
    # Show example usage
    print("\n" + "="*90)
    print("EXAMPLE USAGE")
    print("="*90)
    
    print("""
# Single complaint scoring:
scorer = PriorityScorer()
result = scorer.calculate_priority_score(
    sentiment_score=-0.7,      # Negative sentiment
    image_severity=0.85,       # High severity
    population_density=0.6,    # Medium-high density (normalized)
    overflow_probability=0.9   # Overflow category
)
print(f"Priority: {result['priority_level']}")
print(f"Score: {result['priority_score']:.3f}")

# Batch processing:
df = pd.read_csv('complaints.csv')
scorer = PriorityScorer()
df_prioritized = scorer.process_batch(df)
df_prioritized.to_csv('prioritized.csv', index=False)
    """)
    
    print("\n" + "="*90)
    print("✅ PRIORITY SCORING ENGINE READY!")
    print("="*90)

if __name__ == "__main__":
    main()