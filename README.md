# Resume Screening with NLP

A pipeline that classifies resumes into job categories. It extracts text from PDF resumes with OCR, cleans the text with NLP, and compares classical ML models against a bidirectional LSTM on two public resume datasets.

Built by Hyeran Park and Linqing Zhu as the final project for CISC 3440 (Machine Learning) at Brooklyn College, Fall 2022. The full write-up with methodology, figures, and discussion is in [the project report (PDF)](CISC3440_Resume%20Screening%20Report_HyeranPark_LinqingZhu.pdf).

## What it does

**Part 1: [`_Final_Resume_Screening.ipynb`](_Final_Resume_Screening.ipynb)**
([Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset): 962 resumes, 25 categories)

- **OCR**: renders PDF resumes to images with `pdf2image`, extracts text with Tesseract, and appends the result to the training data. Tesseract was chosen over EasyOCR and Keras-OCR for clean, high-resolution documents.
- **Text cleaning**: regex removal of URLs, mentions, punctuation, and non-ASCII characters, followed by label encoding.
- **KNN**: TF-IDF features (1,500 terms) with `OneVsRestClassifier(KNeighborsClassifier)`, tuned over k=1–30 using 5-fold cross-validation on accuracy and MSE.
- **Bidirectional LSTM**: Keras tokenizer (6,000-word vocabulary), 64-dimensional embedding, a BiLSTM layer, and dense tanh and softmax layers, trained on an 80/10/10 train/validation/test split.
- **Category grouping**: re-runs both models with the 25 categories grouped into Tech, Non-Tech, and Engineering.

**Part 2: [`_Final_HTML_Reumse_Dateset.ipynb`](_Final_HTML_Reumse_Dateset.ipynb)**
([Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset): 2,400+ LiveCareer resumes in HTML)

- Extracts experience and skills sections from the HTML with BeautifulSoup and drops under-represented categories.
- Compares `CountVectorizer` against `TfidfVectorizer`, each with and without spaCy/NLTK stop-word removal, stemming, and lemmatization.
- Trains KNN, logistic regression, SVC, and random forest classifiers (one-vs-rest) on each combination.

## Results

| Experiment | Model | Test accuracy |
|---|---|---|
| Part 1, 25 categories | BiLSTM (tanh) | **95.9%** |
| Part 1, 25 categories | BiLSTM (ReLU) | 91.8% |
| Part 1, 25 categories | KNN (TF-IDF, k=5) | 98.4% |
| Part 2, 21 categories | SVC (TF-IDF) | ~61–63% |
| Part 2, 21 categories | Logistic regression (TF-IDF + text processing) | 61.1% |

The BiLSTM numbers come from the runs recorded in the report. The notebook committed here holds a later re-run of the same model that scored 79.4%, which shows how sensitive the model was to the random shuffle and initialization.

The Part 1 KNN accuracy (about 99% at k=1–2) is high enough that the report flags possible data leakage. The much lower Part 2 scores on a larger, noisier dataset are the more realistic baseline.

**Known issue:** in the grouped-category experiment, the test cell evaluates the first model (`model`) instead of the grouped model (`model2`). That bug explains the near-zero test accuracy reported for that experiment.

## Stack

Python · Tesseract (pytesseract) · pdf2image · OpenCV · BeautifulSoup · NLTK · spaCy · scikit-learn · TensorFlow/Keras · pandas · Matplotlib/Seaborn · Google Colab

## Running it

The notebooks were written for Google Colab (Python 3.8, pandas 1.x) and read their data from Google Drive.

1. Download both Kaggle datasets linked above.
2. In Google Drive, create the layout the notebooks expect:
   ```
   MyDrive/CISC3440_Project/
   ├── Resume_CSV/ResumeDataSet.csv   # Part 1 dataset
   ├── Resume_CSV/Resume.csv          # Part 2 dataset
   └── Resume_Data/*.pdf              # sample PDF resumes for the OCR step
   ```
3. Open a notebook in Colab and choose **Runtime → Run all**. The first cells install Tesseract, Poppler, and the Python dependencies.

To run locally instead, install Tesseract and Poppler, then `pip install -r requirements.txt`, and replace the `drive.mount(...)` cells and file paths with local paths. The notebooks call `DataFrame.append`, which pandas 2.0 removed, so the requirements pin pandas below 2.
