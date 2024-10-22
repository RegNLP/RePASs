# RePASs: Regulatory Passage Answer Stability Score

## Overview

**RePASs** (Regulatory Passage Answer Stability Score) is an evaluation metric designed to assess the stability and reliability of answers generated from regulatory passages. Leveraging advanced Natural Language Processing (NLP) techniques, RePASs facilitates compliance through automated information retrieval and answer generation, ensuring that generated answers are both accurate and consistent with the source regulations.


## Repository Structure
```bash
RePASs/
├── data/
│   └── ObligationClassificationDataset.json
├── models/
│   └── obligation-classifier-legalbert/  # Saved fine-tuned model
├── scripts/
│   ├── train_model.py
│   └── evaluate_model.py
├── requirements.txt
├── README.md
├── .gitignore
```

## Explanation:
- data/: Contains your dataset files.
- models/: Stores the fine-tuned models.
- scripts/: Contains Python scripts for training and evaluation.
- requirements.txt: Lists all Python dependencies.
- README.md: Provides an overview and instructions.
- .gitignore: Specifies files/folders to ignore in Git.

## Setup Instructions
1. Clone the Repository
```bash
git clone https://github.com/RegNLP/RePASs.git
cd RePASs
```
2. Install Dependencies
It's recommended to use a virtual environment to manage dependencies.
```bash
# Using virtualenv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
3. Prepare Your Input Data
You do **not** need to add the `ObligationClassificationDataset.json` as it is already included in the `data/` directory. Prepare your own input JSON file with the following structure:
```json
[
    {
        "QuestionID": "Q1",
        "RetrievedPassages": ["Passage 1 text.", "Passage 2 text."],
        "Answer": "Generated answer text."
    },
    
]
```
  - QuestionID: A unique identifier for each question.
  - RetrievedPassages: A list of passages retrieved relevant to the question.
  - Answer: The answer generated based on the retrieved passages.

4. Training the Model
Run the training script to fine-tune LegalBERT on the provided dataset.
```bash
python scripts/train_model.py
```
The fine-tuned model and tokenizer will be saved in the `models/obligation-classifier-legalbert/` directory.

5. Evaluating with RePASs
Run the evaluation script by providing your input JSON file and a group method name. The script will generate a results text file summarizing the evaluation metrics.
```bash
python scripts/evaluate_model.py --input_file ./data/sample.json --group_method_name my_method
```
- --input_file: Path to your input JSON file.
- --group_method_name: A name to group and label your results.

The results will be saved in `data/my_method` folder.

## Usage Example
Here's a step-by-step example of how to use RePASs:

1. Train the Model:

```bash
python scripts/train_model.py
```
2. Prepare Input Data (`data/sample.json`):

```json
[
    {
        "QuestionID": "Q1",
        "RetrievedPassages": ["Regulatory passage one.", "Regulatory passage two."],
        "Answer": "This is the generated answer based on the retrieved passages."
    },
    {
        "QuestionID": "Q2",
        "RetrievedPassages": ["Another regulatory passage."],
        "Answer": "Another generated answer."
    }
]
```
3. Evaluate:

```bash
python scripts/evaluate_model.py --input_file ./data/sample.json --group_method_name my_method
```
Output:

```yaml
Processing 1/2: QuestionID ['Q1']
Processing 2/2: QuestionID ['Q2']
Average Entailment Score: 0.113
Average Contradiction Score: 0.095
Average Obligation Coverage Score: 0.800
Average Final Composite Score: 0.605
```
Results are saved in `data/my_method` folder.



## Citation

If you use RePASs in your research or applications, please cite the following paper:

```bibtex
@misc{gokhan2024regnlpactionfacilitatingcompliance,
      title={RegNLP in Action: Facilitating Compliance Through Automated Information Retrieval and Answer Generation}, 
      author={Tuba Gokhan and Kexin Wang and Iryna Gurevych and Ted Briscoe},
      year={2024},
      eprint={2409.05677},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.05677}, 
}
```

## Contact
For any questions or suggestions, please open an issue or contact <a href="mailto:regnlp2025@gmail.com<">
regnlp2025@gmail.com</a>
