# AI Text Detection Project

## Overview

This project addresses the critical challenge of distinguishing between human-written and AI-generated text, a growing necessity in an era dominated by advanced language models. Leveraging state-of-the-art Natural Language Processing (NLP) techniques, this repository presents a comprehensive solution for AI text detection, exploring both sophisticated transformer-based models and a robust Convolutional Neural Network (CNN) approach. The goal is to develop highly accurate and efficient models capable of identifying machine-generated content with high confidence.

## Key Features & Highlights

*   **Multi-Model Approach**: Implemented and evaluated three distinct deep learning architectures:
    *   **RoBERTa-base**: A powerful transformer model fine-tuned for classification.
    *   **RoBERTa-large**: A larger, more complex transformer for enhanced performance.
    *   **Custom CNN**: A tailored Convolutional Neural Network designed for text feature extraction.
*   **Robust Data Pipeline**: Utilized a substantial dataset from Kaggle (`human-vs-llm-text-corpus`), meticulously preprocessed and split for rigorous training, validation, and testing.
*   **Performance Optimization**: Employed techniques such as gradient accumulation, mixed-precision training (FP16), and strategic layer freezing to optimize training efficiency and model performance.
*   **Comprehensive Evaluation Metrics**: Models are assessed using industry-standard metrics including Accuracy, Precision, Recall, and F1-Score, ensuring a holistic understanding of their effectiveness.
*   **Modular & Reproducible Codebase**: Notebooks are structured for clarity, ease of understanding, and reproducibility, promoting collaborative development and transparent experimentation.

## Technical Stack

*   **Languages**: Python
*   **Core Libraries**:
    *   `Pandas`, `NumPy`: For efficient data manipulation and numerical operations.
    *   `Hugging Face Transformers`: For leveraging pre-trained RoBERTa models and their tokenization capabilities.
    *   `PyTorch`: Deep learning framework used for RoBERTa model training.
    *   `TensorFlow`/`Keras`: Deep learning framework used for CNN model development and training.
    *   `scikit-learn`: For data splitting and performance metric calculation.
    *   `datasets`: For efficient handling of large datasets with Hugging Face models.
    *   `opendatasets`: For seamless Kaggle dataset integration.
*   **Development Tools**: Jupyter Notebooks, Git, GitHub

## Methodology

### Data Acquisition & Preprocessing

The project utilizes the `human-vs-llm-text-corpus` dataset from Kaggle, comprising text samples categorized as "Human" or "GPT-3.5".
*   Data is loaded from a `.parquet` file.
*   A balanced subset of 26,000 samples from each category (Human and GPT-3.5) is extracted for training.
*   Text labels are converted into numerical format (0 for Human, 1 for AI/GPT-3.5).
*   The dataset is strategically split into training, validation, and test sets to ensure unbiased model evaluation.

### Model Architectures

#### 1. RoBERTa-base for Sequence Classification

*   **Architecture**: `RobertaForSequenceClassification` from the Hugging Face `transformers` library, initialized with `roberta-base` weights.
*   **Tokenizer**: `RobertaTokenizer` with a `max_length` of 1024.
*   **Fine-tuning Strategy**: To enhance efficiency and prevent catastrophic forgetting, the majority of the pre-trained RoBERTa layers were frozen, with only the last two encoder layers and the classification head being fine-tuned on our specific dataset.

#### 2. RoBERTa-large for Sequence Classification

*   **Architecture**: `RobertaForSequenceClassification` from the Hugging Face `transformers` library, utilizing the larger `roberta-large` pre-trained weights.
*   **Tokenizer**: `RobertaTokenizer` with a `max_length` of 512.
*   **Fine-tuning Strategy**: Similar to the base model, fine-tuning was focused on the last two encoder layers and the classification head. This allows for leveraging the more extensive pre-trained knowledge of RoBERTa-large while adapting it to the binary classification task.

#### 3. Custom Convolutional Neural Network (CNN)

*   **Architecture**: A custom-built CNN model implemented using TensorFlow/Keras.
    *   `TextVectorization` layer: Converts raw text into numerical sequences, with a vocabulary size of 5000 and a sequence length of 1000.
    *   `Embedding` layer: Transforms numerical input into dense vector representations.
    *   Multiple `Conv1D` layers: Extract local features from the text.
    *   `BatchNormalization` and `Dropout` layers: Applied throughout the network to improve training stability and prevent overfitting.
    *   `MaxPooling1D` layers: Downsample feature maps, reducing dimensionality.
    *   `Flatten` layer: Prepares the output for the dense classification layers.
    *   `Dense` layers: Final classification layers with a sigmoid activation for binary output.
*   **Training**: Compiled with the `Adam` optimizer and `binary_crossentropy` loss.

### Training & Evaluation

All models were trained with a focus on optimizing performance and resource utilization:
*   **Batch Processing**: Training and evaluation were performed using appropriate batch sizes (e.g., 8 for RoBERTa models with gradient accumulation, 64 for CNN).
*   **Mixed-Precision Training (FP16)**: Utilized for RoBERTa models to accelerate training and reduce memory consumption on compatible hardware.
*   **Model Checkpointing**: Best performing models were saved based on validation accuracy, ensuring optimal model selection.
*   **Logging**: Detailed training logs were recorded for performance analysis and visualization.

## Performance Highlights

Our models demonstrated strong performance in accurately classifying human-written versus AI-generated text:

*   **RoBERTa-base Model**: Achieved a **test accuracy of ~95.77%** and an **F1-score of ~95.94%**. This model showed excellent generalization capabilities with a high recall of 1.0 on the test set, indicating its effectiveness in identifying positive cases (AI-generated text).
*   **RoBERTa-large Model**: Showed competitive performance with a **test accuracy of ~92.50%** and an **F1-score of ~92.68%**. (Note: The specific training run for this notebook used a smaller subset of data for faster iteration. Performance on the full dataset might vary).
*   **Custom CNN Model**: During its training, the CNN model reached a **validation accuracy of ~94.51%** within a few epochs, demonstrating its potential for robust text classification even with a simpler architecture compared to transformers.

These results highlight the project's success in developing effective tools for AI text detection, capable of robust performance across different model complexities.

## Getting Started

To explore this project:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Venkat-Git98/AI-Text-Discriminator.git
    cd AI-Text-Discriminator/notebooks
    ```
2.  **Install dependencies**: It is recommended to create a new virtual environment.
    ```bash
    # For RoBERTa notebooks
    pip install pandas numpy opendatasets torch datasets scikit-learn transformers
    # For CNN notebook
    pip install pandas numpy tensorflow scikit-26learn opendatasets
    ```
    *(Note: Specific versions might be required for full reproducibility, refer to environment files if available in the main repository.)*
3.  **Download the dataset**: The notebooks will automatically download the dataset from Kaggle if not present. Ensure you have Kaggle API credentials configured if running outside a Kaggle environment.
4.  **Run Jupyter Notebooks**: Open and run the `.ipynb` files in your Jupyter environment.

## Future Enhancements

*   **Expanded Dataset Integration**: Explore and integrate larger, more diverse datasets for training to improve model generalization.
*   **Advanced Fine-tuning Strategies**: Experiment with different layer freezing strategies, learning rate schedules, and optimizers for further performance gains.
*   **Ensemble Modeling**: Combine predictions from multiple models (e.g., RoBERTa and CNN) to achieve even higher accuracy and robustness.
*   **Deployment**: Develop an API or web application for real-time text detection.
*   **Explainability**: Incorporate techniques like LIME or SHAP to understand model predictions.


---
