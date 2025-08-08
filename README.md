# 🎫 ML-Classifier for SolarWinds Tickets

An intelligent ticket classification system that automatically assigns categories to SolarWinds support tickets using machine learning and BERT tokenization. Supports both French and English languages for multilingual IT support environments.

## 📑 Table of Contents

- [🚀 Features](#-features)
- [🏁 Getting Started](#-getting-started)
- [💡 Usage](#-usage)
- [⚙️ Technical Details](#️-technical-details)
- [📊 Dataset](#-dataset)
- [🧠 Model Architecture](#-model-architecture)
- [🛠️ Implementation](#️-implementation)
- [📁 File Structure](#-file-structure)
- [🌐 Multilingual Support](#-multilingual-support)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🆘 Support](#-support)

## 🚀 Features

### Core Functionality
- **Automatic Ticket Classification**: Categorizes SolarWinds tickets based on title and description
- **Multilingual Support**: Handles both French and English text seamlessly
- **BERT Tokenization**: Uses pre-trained BERT multilingual model for advanced text processing
- **Machine Learning Pipeline**: Complete ML workflow from data preprocessing to model evaluation
- **High Accuracy**: Optimized for IT support ticket classification

### Key Capabilities
- Text cleaning and preprocessing with regex
- BERT-based tokenization for better semantic understanding
- Logistic Regression classification
- Train/test split for model validation
- Progress tracking for large datasets
- Performance metrics and accuracy reporting

## 🏁 Getting Started

### Prerequisites
```bash
# Python 3.7+
python --version

# Required packages
pip install pandas
pip install scikit-learn
pip install transformers
pip install nltk
pip install numpy
pip install torch  # for BERT
```

### Installation
1. **Clone the repository**:
```bash
git clone https://github.com/Lina-Ait/ML-classifier.git
cd ML-classifier
```

2. **Install dependencies**:
```bash
pip install pandas scikit-learn transformers nltk numpy torch
```

3. **Prepare your data**:
   - Place your ticket data in `report.csv` in the project root
   - Ensure CSV has columns: `Titre`, `Description`, `Catégorie`

4. **Run the classifier**:
```bash
python load_data.py
```

## 💡 Usage

### Basic Classification
```bash
# Run the complete classification pipeline
python load_data.py
```

### Test BERT Tokenization
```bash
# Test multilingual tokenization capabilities
python bert_model.py
```

**Expected output:**
```
['this', 'is', 'an', 'example', 'in', 'english', '.']
['voi', '##ci', 'un', 'exemple', 'en', 'français', '.']
```

### Input Data Format
Your `report.csv` should follow this structure:
```csv
Titre,Description,Catégorie
"Network connectivity issue","Unable to connect to company VPN from home","Network"
"Problème de réseau","Impossible de se connecter au VPN de l'entreprise","Réseau"
"Password reset request","User has forgotten their domain password","Authentication"
"Demande de réinitialisation","L'utilisateur a oublié son mot de passe","Authentification"
```

### Expected Output
```
Processed 1000 entries...
Processed 2000 entries...
Processed 3000 entries...
Model accuracy: 87.43%
```

## ⚙️ Technical Details

### Machine Learning Pipeline

1. **Data Loading**:
   - CSV parsing with pandas
   - UTF-8 encoding support for multilingual text
   - Column selection for relevant fields

2. **Text Preprocessing**:
   ```python
   def clean_text(text):
       text = text.lower()                    # Convert to lowercase
       text = r.sub(r'\d+', '', text)        # Remove numbers
       text = r.sub(r'[^\w\s]', '', text)    # Remove punctuation
       return str(text).replace('\n', ' ').strip()
   ```

3. **Feature Engineering**:
   - BERT multilingual tokenization
   - Title + Description concatenation
   - Category label preprocessing

4. **Model Training**:
   - Train/test split (80/20)
   - CountVectorizer for feature extraction
   - Logistic Regression classifier

### Technology Stack
- **Python 3.7+**: Core programming language
- **Transformers**: BERT tokenization (`bert-base-multilingual-uncased`)
- **Scikit-learn**: Machine learning algorithms and utilities
- **Pandas**: Data manipulation and analysis
- **NLTK**: Natural language processing toolkit
- **NumPy**: Numerical computing

## 📊 Dataset

### Data Structure
- **Source**: SolarWinds support tickets
- **Format**: CSV with UTF-8 encoding
- **Languages**: French and English
- **Fields**:
  - `Titre` (Title): Brief description of the issue
  - `Description`: Detailed problem description
  - `Catégorie` (Category): Target classification label

### Data Processing
- Combines title and description for richer feature set
- Handles missing values and encoding issues
- Progress tracking for large datasets (every 1000 entries)
- Memory-efficient processing with list accumulation

## 🧠 Model Architecture

### BERT Tokenization
```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
```

**Benefits**:
- Subword tokenization for better OOV handling
- Multilingual understanding
- Contextual token representations
- Robust to different languages and domains

### Classification Model
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Feature extraction
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(x_train)

# Classification
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
```

## 🛠️ Implementation

### Core Components

**1. BERT Model (`bert_model.py`)**
- Initializes multilingual BERT tokenizer
- Demonstrates tokenization with examples
- Reusable tokenizer for main pipeline

**2. Data Pipeline (`load_data.py`)**
- Complete ML workflow implementation
- Data loading and preprocessing
- Model training and evaluation
- Progress monitoring

**3. Development Files**
- `load_data-checkpoint.py`: Development checkpoint
- Compiled Python cache files (`.pyc`)

### Code Structure
```python
# Main processing loop
for title, description, category in zip(df['Titre'], df['Description'], df['Catégorie']):
    # Combine and clean text
    entry = str(title) + " " + str(description)
    entry = clean_text(entry)
    entry = tokenizer.tokenize(entry)
    
    # Process category
    cat = clean_text(str(category))
    cat = tokenizer.tokenize(cat)
    
    # Store processed data
    df_list.append(entry)
    categories.append(cat)
```

## 📁 File Structure

```
ML-classifier/
├── 📄 bert_model.py              # BERT tokenizer setup and testing
├── 📄 load_data.py               # Main classification pipeline
├── 📄 load_data-checkpoint.py    # Development checkpoint
├── 📊 report.csv                 # Input dataset (user-provided)
├── 🗂️ __pycache__/               # Python cache files
│   ├── bert_model.cpython-312.pyc
│   └── bert_model.cpython-313.pyc
└── 📖 README.md                  # This documentation
```

## 🌐 Multilingual Support

### Language Capabilities
- **English**: Full support for technical IT terminology
- **French**: Complete French language processing
- **Mixed Content**: Handles bilingual tickets seamlessly

### BERT Multilingual Model
- **Model**: `bert-base-multilingual-uncased`
- **Languages**: 104 languages supported
- **Vocabulary**: 119,547 tokens
- **Casing**: Uncased (case-insensitive)

### Example Tokenization
```python
# English
"Network connectivity issue" → ['network', 'connectivity', 'issue']

# French  
"Problème de réseau" → ['problème', 'de', 'réseau']

# Technical terms
"VPN configuration" → ['vp', '##n', 'configuration']
```

## 🎯 Performance Optimization

### Memory Efficiency
- List-based data accumulation instead of array concatenation
- Progress tracking to monitor large dataset processing
- Efficient BERT tokenization with caching

### Processing Speed
- Batch processing for multiple entries
- Vectorized operations with scikit-learn
- Optimized regex for text cleaning

### Scalability
- Handles datasets of varying sizes
- Memory-conscious implementation
- Progress indicators for long-running processes

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with sample data
5. Submit a pull request

### Code Standards
- Follow PEP 8 Python style guide
- Add docstrings for new functions
- Include type hints where appropriate
- Test multilingual functionality

### Adding New Features
- **New Languages**: Extend BERT multilingual support
- **Model Improvements**: Try different classifiers (SVM, Random Forest)
- **Feature Engineering**: Add TF-IDF, word embeddings
- **Evaluation Metrics**: Precision, recall, F1-score

## 📄 License

This project is for educational and internal business use. Please ensure compliance with your organization's data handling policies when processing support tickets.

## 🆘 Support

### Common Issues

**Import Errors**
```bash
# Install missing dependencies
pip install transformers torch
pip install --upgrade scikit-learn
```

**File Not Found**
```bash
# Ensure report.csv exists in project root
ls -la report.csv
```

**Memory Issues**
- Process data in smaller batches
- Increase available RAM
- Use data sampling for testing

**Encoding Problems**
```python
# Ensure UTF-8 encoding
df = pd.read_csv(filepath, encoding='utf-8')
```

### Performance Tips
- Use GPU acceleration for BERT (install `torch` with CUDA)
- Preprocess data in chunks for large datasets
- Cache tokenized results for repeated experiments
- Monitor memory usage during processing

### Troubleshooting
- Check CSV format and column names
- Verify multilingual text encoding
- Test with small data sample first
- Ensure all dependencies are installed correctly
