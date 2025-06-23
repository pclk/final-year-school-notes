# Glassdoor Job Data Token Analysis Plan

## 1. Initial Setup
### 1.1 Data Loading
- Load CSV file
- Verify required columns presence:
  - query
  - country
  - job_description
  - location
  - salary
  - job_title
  - job_link

### 1.2 Configuration Setup
- Initialize NLTK resources
- Set up stopwords for English
- Compile regex patterns for word detection
- Configure text preprocessing steps

### 1.3 Token Analysis Configuration
- Document template structure
- Establish token counting methodology using tiktoken
- Calculate base template tokens
- Set up Claude API cost constants:
  - Input cost ($0.4/1M tokens)
  - Output cost ($2.0/1M tokens)

## 2. Original Text Analysis
### 2.1 Input Token Calculation
- Calculate tokens for each entry:
  - Template tokens (fixed cost)
  - Job title tokens
  - Location tokens
  - Salary tokens
  - Description tokens
- Track total input tokens

### 2.2 Output Token Estimation
- Calculate expected output tokens:
  - JSON structure tokens
  - Extracted features tokens
  - Standard response format tokens
- Track total output tokens

### 2.3 Cost Analysis
- Calculate total input token cost
- Calculate total output token cost
- Determine total API cost for original text

## 3. Filtered Text Processing
### 3.1 Text Filtering Process
- Remove stopwords from:
  - Job titles
  - Locations
  - Job descriptions
- Preserve salary information

### 3.2 Filtered Token Calculation
- Recalculate tokens for filtered text:
  - Template tokens (unchanged)
  - Filtered job title tokens
  - Filtered location tokens
  - Original salary tokens
  - Filtered description tokens
- Track total filtered input tokens

### 3.3 Filtered Cost Analysis
- Calculate filtered input token cost
- Calculate filtered output token cost
- Determine total API cost for filtered text

## 4. Analysis and Results
### 4.1 Token Reduction Analysis
- Calculate token reduction:
  - Total token difference
  - Percentage reduction
  - Per-field reduction metrics

### 4.2 Cost Savings Analysis
- Calculate cost differences:
  - Input cost savings
  - Output cost savings
  - Total cost savings
- Determine cost-effectiveness

### 4.3 export as clean_csv
