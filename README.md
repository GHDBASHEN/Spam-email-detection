# Spam Email Detection System

This project is a machine learning system designed to detect spam emails using natural language processing (NLP) techniques and the Naive Bayes algorithm to classify messages as either **spam** or **ham (non-spam)**.

![{EBFBB93F-D0C6-4EC8-878A-762E6B4E4A75}](https://github.com/user-attachments/assets/d1b7786c-92f5-4b94-af05-5e97fd514cbd)


## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Features
- Loads and processes a labeled dataset of email messages.
- Converts text into feature vectors using `CountVectorizer`.
- Applies the Naive Bayes algorithm for spam classification.
- Provides accuracy and a detailed classification report for evaluation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ghdbashen/spam-email-detection.git
   cd spam-email-detection
   ```
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn
   ```

## Usage
1. Place the `spam.csv` file in the project directory.
2. Run the detection script:
   ```bash
   python Spam_Detection.py
   ```
3. The script outputs accuracy and classification metrics.

## Dataset
The dataset includes:
- Column 1: Labels (`ham` or `spam`)
- Column 2: Corresponding email messages

The model maps `ham` to `0` and `spam` to `1`.

## Technologies
- **Programming Language**: Python
- **Libraries**: pandas, scikit-learn

## Model Performance
Example performance on a test set:
- **Accuracy**: 98%
- Detailed metrics provided in the classification report.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License.

