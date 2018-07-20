# Linguistic Style-Transfer

Neural network model to disentangle and transfer linguistic style in text

---

## Prerequistites

* Python >= 3.x

### Install required Python packages

```bash
pip install -r requirements.txt
```

---

## Pretraining

### Run a corpus cleaner/adapter

```bash
PYTHONPATH=${PROJECT_DIR_PATH} \
python linguistic_style_transfer_model/corpus_adapters/${CORPUS_ADAPTER_SCRIPT}.py
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

### Train Kneser-Ney Language Model
```bash
./scripts/run_language_model_training.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--model-save-path ${LANGUAGE_MODEL_PATH}
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

---

## Visualizations

### Plot validation accuracy metrics

```bash
./scripts/run_validation_scores_visualization_generator.sh \
--saved-model-path ${SAVED_MODEL_PATH}
```

### Plot T-SNE embedding spaces

```bash
./scripts/run_tsne_visualization_generator.sh \
--saved-model-path ${SAVED_MODEL_PATH}
```

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
--use-kenlm \
--language-model-path ${LANGUAGE_MODEL_PATH} \
--generated-text-file-path ${GENERATED_TEXT_FILE_PATH}
```
