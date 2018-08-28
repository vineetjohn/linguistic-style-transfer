# Linguistic Style-Transfer

Neural network model to disentangle and transfer linguistic style in text

---

## Prerequistites

* [python3](https://www.python.org/downloads)
* [pip](https://pip.pypa.io/en/stable/installing/)
* [tensorflow1.x](https://pypi.org/project/tensorflow/)
* [numpy](https://pypi.org/project/numpy/)
* [scipy](https://pypi.org/project/scipy/)
* [nltk](https://pypi.org/project/nltk/)
* [spacy](https://pypi.org/project/spacy/)
* [gensim](https://pypi.org/project/gensim/)
* [kenlm](https://github.com/kpu/kenlm)
* [matplotlib](https://pypi.org/project/matplotlib/)
* [scikit-learn](https://pypi.org/project/scikit-learn/)

---

## Notes

* Ignore `CUDA_DEVICE_ORDER="PCI_BUS_ID"`, `CUDA_VISIBLE_DEVICES="0"` unless you're training with a GPU
* Input data file format: 
    * `${TEXT_FILE_PATH}` should have 1 sentence per line. 
    * Similarly, `${LABEL_FILE_PATH}` should have 1 label per line.

---

## Data Sources

* Yelp Service Reviews - [Link](https://github.com/shentianxiao/language-style-transfer)
* Amazon Product Reviews - [Link](https://github.com/fuzhenxin/text_style_transfer)

---

## Pretraining


### Run a corpus cleaner/adapter

```bash
./scripts/run_corpus_adapter.sh \
linguistic_style_transfer_model/corpus_adapters/${CORPUS_ADAPTER_SCRIPT}
```


### Train word embedding model
```bash
./scripts/run_word_vector_training.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--model-file-path ${WORD_EMBEDDINGS_PATH}
```


### Train validation classifier

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_classifier_training.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--label-file-path ${TRAINING_LABEL_FILE_PATH} \
--training-epochs ${NUM_EPOCHS} --vocab-size ${VOCAB_SIZE}
```

This will produce a folder like `saved-models-classifier/xxxxxxxxxx`.


### Train Kneser-Ney Language Model
```bash
./scripts/run_language_model_training.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--model-save-path ${LANGUAGE_MODEL_PATH}
```


### Extract label-correlated words 
```bash
./scripts/run_word_retriever.sh \
--text-file-path ${TEXT_FILE_PATH} \
--label-file-path ${LABEL_FILE_PATH} \
--logging-level ${LOGGING_LEVEL}
```


---


## Style Transfer Model Training


### Train style transfer model

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_linguistic_style_transfer_model.sh \
--train-model \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--label-file-path ${TRAINING_LABEL_FILE_PATH} \
--training-embeddings-file-path ${TRAINING_WORD_EMBEDDINGS_PATH} \
--validation-text-file-path ${VALIDATION_TEXT_FILE_PATH} \
--validation-label-file-path ${VALIDATION_LABEL_FILE_PATH} \
--validation-embeddings-file-path ${VALIDATION_WORD_EMBEDDINGS_PATH} \
--classifier-saved-model-path ${CLASSIFIER_SAVED_MODEL_PATH} \
--dump-embeddings \
--training-epochs ${NUM_EPOCHS} \
--vocab-size ${VOCAB_SIZE} \
--logging-level="DEBUG"
```

This will produce a folder like `saved-models/xxxxxxxxxx`. 
It will also produce `output/xxxxxxxxxx-training` if validation is turned on.


### Infer style transferred sentences

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_linguistic_style_transfer_model.sh \
--generate-novel-text \
--evaluation-text-file-path ${TEST_TEXT_FILE_PATH} \
--saved-model-path ${SAVED_MODEL_PATH} \
--logging-level="DEBUG"
```

This will produce a folder like `output/xxxxxxxxxx-inference`.


---


## Visualizations


### Plot validation accuracy metrics

```bash
./scripts/run_validation_scores_visualization_generator.sh \
--saved-model-path ${SAVED_MODEL_PATH}
```

This will produce a few files like `${SAVED_MODEL_PATH}/validation_xxxxxxxxxx.svg`


### Plot T-SNE embedding spaces

```bash
./scripts/run_tsne_visualization_generator.sh \
--saved-model-path ${SAVED_MODEL_PATH}
```

This will produce a few files like `${SAVED_MODEL_PATH}/tsne_plots/tsne_embeddings_plot_xx.svg`


---


## Run evaluation metrics


### Style Transfer

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_style_transfer_evaluator.sh \
--classifier-saved-model-path ${CLASSIFIER_SAVED_MODEL_PATH} \
--text-file-path ${TEST_TEXT_FILE_PATH} \
--label-index ${TEST_TEXT_FILE_LABEL}
```


### Content Preservation

```bash
./scripts/run_content_preservation_evaluator.sh \
--embeddings-file-path ${VALIDATION_WORD_EMBEDDINGS_PATH} \
--source-file-path ${TEST_TEXT_FILE_PATH} \
--target-file-path ${GENERATED_TEXT_FILE_PATH}
```


### Latent Space Classification Accuracy

```bash
./scripts/run_classifier_prediction.sh \
--gold-labels-file-path ${TEST_LABEL_FILE_PATH} \
--saved-model-path ${SAVED_MODEL_PATH} \
--predictions-file-path ${PREDICTIONS_LABEL_FILE_PATH}
```


### Language Fluency

```bash
./scripts/run_language_fluency_evaluator.sh \
--language-model-path ${LANGUAGE_MODEL_PATH} \
--generated-text-file-path ${GENERATED_TEXT_FILE_PATH}
```


### All Evaluation Metrics (works only for the output of this project)

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_all_evaluators.sh \
--embeddings-path ${VALIDATION_WORD_EMBEDDINGS_PATH} \
--language-model-path ${LANGUAGE_MODEL_PATH} \
--classifier-model-path ${CLASSIFIER_SAVED_MODEL_PATH} \
--training-path ${SAVED_MODEL_PATH} \
--inference-path ${GENERATED_SENTENCES_SAVE_PATH}
```
