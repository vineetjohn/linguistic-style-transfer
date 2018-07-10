from random import shuffle

from linguistic_style_transfer_model.config import human_annotation_config

annotation_cfg = human_annotation_config.annotation_config


def get_samples(original_file_path, generated_file_path, num_samples):
    with open(original_file_path) as original_file:
        original_sentences = original_file.readlines()
    with open(generated_file_path) as generated_file:
        generated_sentences = generated_file.readlines()

    num_available = len(original_sentences)
    indices = list(range(num_available))
    shuffle(indices)
    selected_indices = indices[:num_samples]

    selected_samples = list()
    for index in selected_indices:
        selected_samples.append((original_sentences[index].strip(), generated_sentences[index].strip()))

    return selected_samples


def main():
    annotation_file_path = human_annotation_config.output_folder + "manual_annotation.tsv"
    mapping_file_path = human_annotation_config.output_folder + "mapping_file.tsv"

    total_count = 0
    for model_id in annotation_cfg:
        model_cfg = annotation_cfg[model_id]
        total_count += model_cfg['count']

    print("total_count: {}".format(total_count))
    indices = list(range(total_count))
    shuffle(indices)
    # print(indices)
    indices_to_model = dict()
    tabular_samples = list()

    start = 0
    for model_id in annotation_cfg:
        model_cfg = annotation_cfg[model_id]
        num_samples = model_cfg['count']
        end = start + num_samples
        rand_ids = indices[start:end]
        for rand_id in rand_ids:
            indices_to_model[rand_id] = model_id
        # print(rand_ids)

        attributes = model_id.split("-")
        dataset = attributes[1]
        type = "-".join(attributes[2:])

        samples = get_samples(model_cfg['original'], model_cfg['generated'], num_samples)
        assert num_samples == len(samples)

        for i, (original, generated) in enumerate(samples):
            tabular_samples.append(
                [str(rand_ids[i]), dataset, type, original, generated])
        start = end

    with open(annotation_file_path, 'w') as annotation_file:
        annotation_file.write("id\tdataset\ttype\toriginal\tgenerated\n")
        shuffle(tabular_samples)
        for sample in tabular_samples:
            annotation_file.write("\t".join(sample) + "\n")
        print("Printed annotation file")

    with open(mapping_file_path, 'w') as mapping_file:
        for index in indices_to_model:
            mapping_file.write("{}\t{}\n".format(index, indices_to_model[index]))
        print("Printed mapping file")
    # print("indices_to_model: {}".format(indices_to_model))


if __name__ == '__main__':
    main()
