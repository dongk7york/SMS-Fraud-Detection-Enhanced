# Presentation Charts

This folder contains all visualization charts for the SMS Fraud Detection project presentation.

## Charts Overview

### 1. Preprocessing Pipeline
**File:** `preprocessing_pipeline.png`
- **Purpose:** Explains text preprocessing with POS-aware lemmatization
- **Content:**
  - Step-by-step preprocessing flow
  - POS tagging explanation
  - Comparison: POS-aware vs simple lemmatization
  - Real examples at each step
- **Use in slides:** Introduction to data preprocessing

### 2. TF-IDF Explanation (4 charts)

#### 2.1 Main Concept
**File:** `tfidf_main_concept.png`
- **Purpose:** Core TF-IDF concept and formula
- **Content:**
  - What TF-IDF measures
  - TF × IDF formula breakdown
  - High vs low score interpretation
- **Use in slides:** Introducing TF-IDF

#### 2.2 Example Calculation
**File:** `tfidf_example_calculation.png`
- **Purpose:** Concrete calculation example
- **Content:**
  - 3 SMS messages corpus
  - Step-by-step TF calculation
  - Step-by-step IDF calculation
  - Final TF-IDF score
- **Use in slides:** Making TF-IDF concrete

#### 2.3 Matrix Output
**File:** `tfidf_matrix_output.png`
- **Purpose:** Show TF-IDF vectorization output
- **Content:**
  - Sparse matrix representation
  - Example matrix values
  - Memory efficiency explanation
- **Use in slides:** Explaining feature extraction

#### 2.4 Advantages & Limitations
**File:** `tfidf_advantages_limitations.png`
- **Purpose:** Critical analysis of TF-IDF
- **Content:**
  - Advantages (distinctive words, simple, etc.)
  - Limitations (no semantics, bag-of-words, etc.)
  - Solution: Combine with Word2Vec
- **Use in slides:** Motivating enhanced approach

### 3. Word2Vec Architecture (3 charts)

#### 3.1 Simplified Architecture
**File:** `word2vec_simplified_architecture.png`
- **Purpose:** Explain Word2Vec in simple terms
- **Content:**
  - The problem: converting words to meaningful numbers
  - 3-step solution: Input → Neural Network → Word Embedding
  - Key insight: similar contexts → similar vectors
- **Use in slides:** Introducing Word2Vec concept

#### 3.2 Training Flow
**File:** `word2vec_training_flow.png`
- **Purpose:** Show how Word2Vec learns
- **Content:**
  - Training example: "send text message"
  - Input layer (one-hot encoding)
  - Hidden layer (learns patterns)
  - Output layer (predicts context)
  - Extraction of embeddings
- **Use in slides:** Explaining the learning process

#### 3.3 Distributional Hypothesis
**File:** `word2vec_distributional_hypothesis.png`
- **Purpose:** Explain why Word2Vec works
- **Content:**
  - "You shall know a word by the company it keeps"
  - Words in similar contexts have similar meanings
  - Example: "text" and "message" both appear with "send"
- **Use in slides:** Theoretical foundation

### 4. Word2Vec Visualization
**File:** `word2vec_2d_visualization.png`
- **Purpose:** Show semantic clustering learned by Word2Vec
- **Content:**
  - 4 semantic clusters with convex hulls
  - Communication Methods: send, text, phone, number, call
  - Spam/Prize: winner, claim, prize
  - Communication Actions: reply, message, stop
  - Urgency: urgent, contact
  - 2 contrast words: receive, make
- **Use in slides:** Demonstrating Word2Vec semantic understanding

### 5. Model Comparison
**File:** `model_comparison.png`
- **Purpose:** Show performance improvement
- **Content:**
  - Baseline vs Enhanced model comparison
  - Metrics: Accuracy, Precision, Recall, F1-Score
  - Visual bar charts
- **Use in slides:** Results and conclusions

## Presentation Flow Suggestion

1. **Introduction**
   - Problem statement
   - Dataset overview

2. **Data Preprocessing**
   - Show: `preprocessing_pipeline.png`
   - Explain POS-aware lemmatization

3. **Feature Extraction: TF-IDF**
   - Show: `tfidf_main_concept.png`
   - Show: `tfidf_example_calculation.png`
   - Show: `tfidf_matrix_output.png`
   - Show: `tfidf_advantages_limitations.png`

4. **Feature Extraction: Word2Vec**
   - Show: `word2vec_simplified_architecture.png` (concept)
   - Show: `word2vec_training_flow.png` (how it learns)
   - Show: `word2vec_distributional_hypothesis.png` (why it works)
   - Show: `word2vec_2d_visualization.png` (results)

5. **Results**
   - Show: `model_comparison.png`
   - Discuss performance improvements

6. **Conclusions**
   - Summary of findings
   - Future work

## Technical Details

- **Resolution:** All charts are 300 DPI (high quality for printing)
- **Format:** PNG with white background
- **Size:** Optimized for 16:9 presentation slides
- **Color Scheme:** Consistent across all charts
  - Input: Light blue (#e1f5fe)
  - Process: Light orange (#fff3e0)
  - Output: Light green (#c8e6c9)
  - Special: Light purple (#f3e5f5)

## Regenerating Charts

### Preprocessing Chart
```bash
python3 generate_preprocessing_flowchart.py
```

### TF-IDF Charts
```bash
python3 convert_mermaid_to_png.py
```

### Word2Vec Visualization
```bash
python3 visualize_word2vec.py
```

### Model Comparison
Generated automatically during model training in `sms_fraud_enhanced.py`

## Notes

- All charts are ready to insert directly into PowerPoint, Google Slides, or Keynote
- Charts maintain consistent styling for professional appearance
- High resolution ensures clarity when projected
- White backgrounds work well with most presentation themes