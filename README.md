# Linguistic Style-Transfer

Neural network model to disentangle and transfer linguistic style in text

---

## Run a corpus cleaner/adapter

```bash
export PYTHONPATH=${PROJECT_DIR_PATH} && \
python linguistic_style_transfer_model/corpus_adapters/${CORPUS_ADAPTER_SCRIPT}.py
```

## Train word embedding model
```bash
./run_word_vector_training.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--model-file-path ${WORD_EMBEDDINGS_PATH}
```

## Train validation classifier

```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID" && \
export CUDA_VISIBLE_DEVICES="0" && \
export TF_CPP_MIN_LOG_LEVEL=1 && \
./run_classifier_training.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--label-file-path ${TRAINING_LABEL_FILE_PATH} \
--training-epochs ${NUM_EPOCHS} --vocab-size ${VOCAB_SIZE}
```

---

## Train style transfer model

```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID" && \
export CUDA_VISIBLE_DEVICES="0" && \
export TF_CPP_MIN_LOG_LEVEL=1 && \
./run_linguistic_style_transfer_model.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--label-file-path ${TRAINING_LABEL_FILE_PATH} \
--training-embeddings-file-path ${TRAINING_WORD_EMBEDDINGS_PATH} \
--validation-text-file-path ${VALIDATION_TEXT_FILE_PATH} \
--validation-label-file-path ${VALIDATION_LABEL_FILE_PATH} \
--validation-embeddings-file-path ${VALIDATION_WORD_EMBEDDINGS_PATH} \
--classifier-saved-model-path ${CHECKPOINT_DIR_PATH} \
--train-model \
--dump-embeddings \
--training-epochs ${NUM_EPOCHS} \
--vocab-size ${VOCAB_SIZE} \
--logging-level="DEBUG"
```

## Evaluate style transfer model

```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID" && \
export CUDA_VISIBLE_DEVICES="0" && \
export TF_CPP_MIN_LOG_LEVEL=1 && \
./run_linguistic_style_transfer_model.sh \
--evaluation-text-file-path ${TEST_TEXT_FILE_PATH} \
--saved-model-path ${SAVED_MODEL_PATH} \
--generate-novel-text \
--logging-level="DEBUG"
```

---

## Run evaluation metrics

### Style Transfer

```bash
./run_style_transfer_evaluator.sh \
--checkpoint-dir ${CHECKPOINT_DIR_PATH} \
--text-file-path ${TEST_TEXT_FILE_PATH} \
--label-index ${TEST_TEXT_FILE_LABEL}
```

### Content Preservation

```bash
./run_content_preservation_evaluator.sh \
--embeddings-file-path ${VALIDATION_WORD_EMBEDDINGS_PATH} \
--source-file-path ${TEST_TEXT_FILE_PATH} \
--target-file-path ${GENERATED_TEXT_FILE_PATH}
```
