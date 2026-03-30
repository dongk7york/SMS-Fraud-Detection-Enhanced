"""
Feature Ablation Study: Impact of 14 Engineered Features
=========================================================

This script compares model performance with and without the 14 engineered features
to quantify their contribution to fraud detection accuracy.

Comparison Groups:
1. TF-IDF + Dual Word2Vec (1700 features) - NEW TRAINING
2. TF-IDF + Dual Word2Vec + 14 Engineered Features (1714 features) - FROM baseline_vs_enhanced_results.csv

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Import from main script
import sys
sys.path.append('.')
from sms_fraud_enhanced import (
    SMSPreprocessor, 
    load_and_analyze_data,
    load_uci_data,
    balance_and_combine_datasets,
    train_url_word2vec,
    train_text_word2vec,
    DualWord2VecFeatureExtractor
)


def train_models_without_engineered(X_train, X_test, y_train, y_test):
    """Train models WITHOUT engineered features"""
    results = {}
    
    print(f"\n{'='*80}")
    print(f"Training Models WITHOUT 14 Engineered Features")
    print(f"Feature shape: {X_train.shape} (TF-IDF + Dual Word2Vec only)")
    print(f"{'='*80}")
    
    # Logistic Regression
    print("\n[1/4] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    results['LR'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'f1': f1_score(y_test, y_pred_lr)
    }
    
    print(f"Accuracy: {results['LR']['accuracy']:.4f}")
    print(f"F1-Score: {results['LR']['f1']:.4f}")
    
    # SVM
    print("\n[2/4] Training SVM...")
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    
    results['SVM'] = {
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'precision': precision_score(y_test, y_pred_svm),
        'recall': recall_score(y_test, y_pred_svm),
        'f1': f1_score(y_test, y_pred_svm)
    }
    
    print(f"Accuracy: {results['SVM']['accuracy']:.4f}")
    print(f"F1-Score: {results['SVM']['f1']:.4f}")
    
    # Random Forest
    print("\n[3/4] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    results['RF'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf)
    }
    
    print(f"Accuracy: {results['RF']['accuracy']:.4f}")
    print(f"F1-Score: {results['RF']['f1']:.4f}")
    
    # Gradient Boosting
    print("\n[4/4] Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    
    results['GB'] = {
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb),
        'recall': recall_score(y_test, y_pred_gb),
        'f1': f1_score(y_test, y_pred_gb)
    }
    
    print(f"Accuracy: {results['GB']['accuracy']:.4f}")
    print(f"F1-Score: {results['GB']['f1']:.4f}")
    
    return results


def load_enhanced_results():
    """Load results WITH engineered features from existing CSV"""
    print(f"\n{'='*80}")
    print("Loading Results WITH 14 Engineered Features")
    print("Source: baseline_vs_enhanced_results.csv")
    print(f"{'='*80}")
    
    df = pd.read_csv('baseline_vs_enhanced_results.csv')
    
    # Filter only Enhanced models
    enhanced_df = df[df['Model'].str.startswith('Enhanced_')]
    
    results = {}
    for _, row in enhanced_df.iterrows():
        model_name = row['Model'].replace('Enhanced_', '')
        results[model_name] = {
            'accuracy': row['Accuracy'],
            'precision': row['Precision'],
            'recall': row['Recall'],
            'f1': row['F1-Score']
        }
    
    print("\nLoaded Enhanced Results:")
    for model, metrics in results.items():
        print(f"  {model}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    return results


def create_comparison_dataframe(without_eng, with_eng):
    """Create comparison dataframe"""
    rows = []
    
    for model_name in without_eng.keys():
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            without_val = without_eng[model_name][metric]
            with_val = with_eng[model_name][metric]
            improvement = ((with_val - without_val) / without_val) * 100
            
            rows.append({
                'Model': model_name,
                'Metric': metric.capitalize(),
                'Without_Engineered': without_val,
                'With_Engineered': with_val,
                'Improvement_%': improvement,
                'Absolute_Diff': with_val - without_val
            })
    
    return pd.DataFrame(rows)


def plot_ablation_results(comparison_df):
    """Create visualization comparing with/without engineered features"""
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Feature Ablation Study: Impact of 14 Engineered Features', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Filter data for this metric
        metric_data = comparison_df[comparison_df['Metric'] == metric]
        
        # Prepare data for grouped bar chart
        models = metric_data['Model'].values
        without = metric_data['Without_Engineered'].values
        with_eng = metric_data['With_Engineered'].values
        
        x = np.arange(len(models))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, without, width, label='Without Engineered (1700 features)',
                       color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, with_eng, width, label='With Engineered (1714 features)',
                       color='#4ECDC4', alpha=0.8)
        
        # Customize
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha='center')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_ylim([0.92, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('charts/feature_ablation_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: charts/feature_ablation_comparison.png")
    plt.close()


def print_summary_statistics(comparison_df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Average improvement per metric
    print("\nAverage Improvement by Metric:")
    avg_by_metric = comparison_df.groupby('Metric')['Improvement_%'].mean()
    for metric, improvement in avg_by_metric.items():
        print(f"  {metric:12s}: {improvement:+.3f}%")
    
    # Average improvement per model
    print("\nAverage Improvement by Model:")
    avg_by_model = comparison_df.groupby('Model')['Improvement_%'].mean()
    for model, improvement in avg_by_model.items():
        print(f"  {model:4s}: {improvement:+.3f}%")
    
    # Overall average
    overall_avg = comparison_df['Improvement_%'].mean()
    print(f"\nOverall Average Improvement: {overall_avg:+.3f}%")
    
    # Absolute differences
    print("\nAverage Absolute Improvement by Metric:")
    avg_abs_by_metric = comparison_df.groupby('Metric')['Absolute_Diff'].mean()
    for metric, diff in avg_abs_by_metric.items():
        print(f"  {metric:12s}: {diff:+.6f}")
    
    # Best improvements
    print("\nTop 5 Improvements:")
    top_improvements = comparison_df.nlargest(5, 'Improvement_%')
    for _, row in top_improvements.iterrows():
        print(f"  {row['Model']:4s} - {row['Metric']:10s}: {row['Improvement_%']:+.3f}% (abs: {row['Absolute_Diff']:+.6f})")
    
    # Worst improvements (or degradations)
    print("\nBottom 5 (Smallest Improvements or Degradations):")
    bottom_improvements = comparison_df.nsmallest(5, 'Improvement_%')
    for _, row in bottom_improvements.iterrows():
        print(f"  {row['Model']:4s} - {row['Metric']:10s}: {row['Improvement_%']:+.3f}% (abs: {row['Absolute_Diff']:+.6f})")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("FEATURE ABLATION STUDY: IMPACT OF 14 ENGINEERED FEATURES")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = SMSPreprocessor()
    
    # Load and combine datasets
    print("\n[1/5] Loading datasets...")
    original_df = load_and_analyze_data('data/original_data.csv')
    original_df['content'] = original_df['SMS text'] + ' ' + original_df['Client Sender ID'] + ' ' + original_df['Country']
    original_df['source'] = 'Original'
    original_df = original_df[['content', 'Case', 'source']]
    
    uci_df = load_uci_data('data/UCI_spam_data.csv')
    combined_df = balance_and_combine_datasets(original_df, uci_df)
    
    # Preprocess
    print("\n[2/5] Preprocessing combined dataset...")
    combined_df['cleaned_text'] = combined_df['content'].apply(preprocessor.clean_text)
    combined_df['urls'] = combined_df['content'].apply(preprocessor.extract_urls)
    
    # Create TF-IDF features
    print("\n[3/5] Creating TF-IDF features...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(combined_df['cleaned_text'])
    X_tfidf_dense = X_tfidf.toarray()
    
    print(f"TF-IDF shape: {X_tfidf.shape}")
    
    # Train Word2Vec models
    print("\n[4/5] Training Word2Vec models...")
    url_w2v = train_url_word2vec(combined_df, preprocessor)
    text_w2v = train_text_word2vec(combined_df)
    
    # Extract Word2Vec embeddings
    print("\nExtracting Text Word2Vec embeddings...")
    text_w2v_embeddings = []
    for text in combined_df['cleaned_text']:
        tokens = text.split()
        vectors = []
        for token in tokens:
            if text_w2v is not None and token in text_w2v.wv:
                vectors.append(text_w2v.wv[token])
        if vectors:
            text_w2v_embeddings.append(np.mean(vectors, axis=0))
        else:
            text_w2v_embeddings.append(np.zeros(100))
    
    text_w2v_features = np.array(text_w2v_embeddings)
    print(f"Text Word2Vec features shape: {text_w2v_features.shape}")
    
    print("\nExtracting URL Word2Vec embeddings...")
    url_w2v_embeddings = []
    feature_extractor = DualWord2VecFeatureExtractor(url_w2v, text_w2v)
    
    for text in combined_df['content']:
        urls = preprocessor.extract_urls(text)
        if urls and url_w2v is not None:
            url_vectors = []
            for url in urls:
                url_vec = feature_extractor.get_url_embedding(url, preprocessor)
                url_vectors.append(url_vec)
            url_w2v_embeddings.append(np.mean(url_vectors, axis=0))
        else:
            url_w2v_embeddings.append(np.zeros(100))
    
    url_w2v_features = np.array(url_w2v_embeddings)
    print(f"URL Word2Vec features shape: {url_w2v_features.shape}")
    
    # Create feature set WITHOUT engineered features
    print("\n[5/5] Creating feature set WITHOUT engineered features...")
    X_without_eng = np.hstack([
        X_tfidf_dense,
        text_w2v_features,
        url_w2v_features
    ])
    
    print(f"Feature set WITHOUT engineered: {X_without_eng.shape}")
    print(f"  - TF-IDF: {X_tfidf_dense.shape[1]}")
    print(f"  - Text Word2Vec: {text_w2v_features.shape[1]}")
    print(f"  - URL Word2Vec: {url_w2v_features.shape[1]}")
    print(f"  - Total: {X_without_eng.shape[1]}")
    
    # Prepare labels and split
    y = combined_df['Case'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_without_eng, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models WITHOUT engineered features
    results_without = train_models_without_engineered(X_train, X_test, y_train, y_test)
    
    # Load results WITH engineered features from existing CSV
    results_with = load_enhanced_results()
    
    # Create comparison
    comparison_df = create_comparison_dataframe(results_without, results_with)
    
    # Save results
    comparison_df.to_csv('feature_ablation_results.csv', index=False)
    print("\n✓ Results saved: feature_ablation_results.csv")
    
    # Print detailed comparison
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Print summary statistics
    print_summary_statistics(comparison_df)
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_ablation_results(comparison_df)
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print("1. Check feature_ablation_results.csv for detailed metrics")
    print("2. View charts/feature_ablation_comparison.png for visual comparison")
    print("\nConclusion:")
    overall_avg = comparison_df['Improvement_%'].mean()
    if overall_avg > 0:
        print(f"✓ The 14 engineered features provide an average improvement of {overall_avg:+.3f}%")
    else:
        print(f"✗ The 14 engineered features show an average change of {overall_avg:+.3f}%")


if __name__ == "__main__":
    main()
