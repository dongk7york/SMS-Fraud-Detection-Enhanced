# SMS Fraud Detection - Enhanced Implementation

## Original Work

This project builds upon [TelecomFraudDetection](https://github.com/nsalhab/TelecomFraudDetection) by nsalhab, which implemented SMS fraud detection using TF-IDF vectorization with Logistic Regression and SVM classifiers.

## Our Enhancements

- **Dual Word2Vec Models**: Separate 100-dimensional embeddings for message text and URL patterns to capture semantic relationships
- **POS-Aware Preprocessing**: Context-sensitive lemmatization using part-of-speech tagging for improved text normalization
- **14 Engineered Features**: URL analysis, urgency scoring, financial indicators, and text statistics
- **Combined Dataset Approach**: Integrated original telecom fraud data with UCI SMS Spam Collection for robust training
- **Comprehensive Evaluation**: Baseline vs Enhanced comparison across multiple ML models (LR, SVM, RF, GB)

## Results

Enhanced SVM achieves **99.13% accuracy** with improved recall (96.77%) compared to baseline SVM (99.03% accuracy, 95.15% recall).

## Files

- **sms_fraud_enhanced.py** - Main training script with baseline and enhanced approaches
- **requirements.txt** - Python dependencies
- **CITATIONS.md** - Detailed citations for data sources and methodologies
- **data/** - Original telecom fraud dataset and UCI SMS Spam Collection
- **charts/** - Model comparison and visualization outputs (10 charts)
- **baseline_vs_enhanced_results.csv** - Detailed performance metrics

## Usage

```bash
pip install -r requirements.txt
python sms_fraud_enhanced.py
```

## License

This project retains the original MIT License from nsalhab's work, which permits modification and redistribution with attribution.

## Citations

See [CITATIONS.md](CITATIONS.md) for complete references including:
- Original repository: https://github.com/nsalhab/TelecomFraudDetection
- UCI SMS Spam Collection: https://doi.org/10.24432/C5CC84