# Linguistic Style-Transfer

Neural network model to disentangle and transfer linguistic style in text

## Run a corpus cleaner/adapter
```bash
export PYTHONPATH=${PROJECT_DIR_PATH} && \
python linguistic_style_transfer_model/corpus_adapters/amazon_review_corpus_adapter.py
```

## Run model
```bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID" && export CUDA_VISIBLE_DEVICES="1" && export TF_CPP_MIN_LOG_LEVEL=1 && \
./run_linguistic_style_transfer_model.sh \
--text-file-path ${TRAINING_TEXT_FILE_PATH} \
--label-file-path ${TRAINING_LABEL_FILE_PATH} \
--validation-text-file-path ${VALIDATION_TEXT_FILE_PATH} \
--validation-label-file-path ${VALIDATION_LABEL_FILE_PATH} \
--validation-embeddings-file-path ${VALIDATION_WORD_EMBEDDINGS_PATH} \
--evaluation-text-file_path ${TEST_TEXT_FILE_PATH} \
--classifier-checkpoint-dir ${CHECKPOINT_DIR_PATH} \
--use-pretrained-embeddings --train-model --generate-novel-text --dump-embeddings \
--training-epochs ${NUM_EPOCHS} --vocab-size ${VOCAB_SIZE} --logging-level="DEBUG"
```

## Evaluate model

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
