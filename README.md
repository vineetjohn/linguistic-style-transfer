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
* Assuming that you already have [g++](https://gcc.gnu.org/) and [bash](http://tiswww.case.edu/php/chet/bash/bashtop.html) installed, run the following commands to setup the [kenlm](https://github.com/kpu/kenlm) library properly:
    * `wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz`
    * `mkdir kenlm/build`
    * `cd kenlm/build`
    * `sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev` (to install basic dependencies)
    * Install [Boost](https://www.boost.org/):
        * Download boost_1_67_0.tar.bz2 from [here](https://www.boost.org/users/history/version_1_67_0.html)
        * `tar --bzip2 -xf /path/to/boost_1_67_0.tar.bz2`
    * Install [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page):
        * `export EIGEN3_ROOT=$HOME/eigen-eigen-07105f7124f9`
        * `cd $HOME; wget -O - https://bitbucket.org/eigen/eigen/get/3.2.8.tar.bz2 |tar xj`
        * Go back to the `kenlm/build` folder and run `rm CMakeCache.txt`
    * `cmake ..`
    * `make -j2`

---

## Data Sources

### Customer Review Datasets
* Yelp Service Reviews - [Link](https://github.com/shentianxiao/language-style-transfer)
* Amazon Product Reviews - [Link](https://github.com/fuzhenxin/text_style_transfer)

### Word Embeddings
References to `${VALIDATION_WORD_EMBEDDINGS_PATH}` in the instructions below should be replaced by the path to the file `glove.6B.100d.txt`, which can be downloaded from [here](http://nlp.stanford.edu/data/glove.6B.zip).

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
Use the below command to train a `n`-gram language model (run from the `kenlm/build` folder)
```bash
./bin/lmplz -o ${n} --text ${TRAINING_TEXT_FILE_PATH} > ${LANGUAGE_MODEL_PATH}
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
--transform-text \
--evaluation-text-file-path ${TEST_TEXT_FILE_PATH} \
--saved-model-path ${SAVED_MODEL_PATH} \
--logging-level="DEBUG"
```

This will produce a folder like `output/xxxxxxxxxx-inference`.


### Generate new sentences

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_linguistic_style_transfer_model.sh \
--generate-novel-text \
--saved-model-path ${SAVED_MODEL_PATH} \
--num-sentences-to-generate ${NUM_SENTENCES}
--logging-level="DEBUG"
```

This will produce a folder like `output/xxxxxxxxxx-generation`.

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
--text-file-path ${GENERATED_TEXT_FILE_PATH} \
--label-index ${GENERATED_TEXT_LABEL}
```

Alternatively, if you have a file with the labels, use the below command instead

```bash
CUDA_DEVICE_ORDER="PCI_BUS_ID" \
CUDA_VISIBLE_DEVICES="0" \
TF_CPP_MIN_LOG_LEVEL=1 \
./scripts/run_style_transfer_evaluator.sh \
--classifier-saved-model-path ${CLASSIFIER_SAVED_MODEL_PATH} \
--text-file-path ${GENERATED_TEXT_FILE_PATH} \
--label-file-path ${GENERATED_LABELS_FILE_PATH}
```


### Content Preservation

```bash
./scripts/run_content_preservation_evaluator.sh \
--embeddings-file-path ${VALIDATION_WORD_EMBEDDINGS_PATH} \
--source-file-path ${TEST_TEXT_FILE_PATH} \
--target-file-path ${GENERATED_TEXT_FILE_PATH}
```


### Latent Space Predicted Label Accuracy

```bash
./scripts/run_label_accuracy_prediction.sh \
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

Log-likelihood values are base 10.


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
