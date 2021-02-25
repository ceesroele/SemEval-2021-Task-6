"""Apply influence function,
see: https://github.com/ryokamoi/pytorch_influence_functions
"""

import pytorch_influence_functions as ptif
from simpletransformers.classification import ClassificationModel, MultiLabelClassificationModel
import pipeline_config
from simpletransformers.classification.classification_utils import (
    InputExample
)
from torch.utils.data import DataLoader
import pandas as pd
import pickle
import json
from datetime import datetime
from fragment_utils import (
    fragment_from_text,
    decode,
    calibrate,
    split_in_sentences,
    insert_tags,
    insert_tags_list
)
import torch
from eval import batched_predict
import converter
from label_tool import label_to_symbol, symbol_to_label

logger = pipeline_config.logger

from score import get_labels, prediction_to_labels

# Split input in sentences, rather than taking it as a single input vector
SPLIT_IN_SENTENCES = True  # False = 0.44, True = 0.41


def get_all_dirty_labels(s):
    """Get any label from an outcome, irrespective of being matched or not"""

    return []

class InfluenceFunction(ClassificationModel):
    def __init__(self, model_type, model_name):
        super(InfluenceFunction, self).__init__(model_type, model_name)
        self.config = pipeline_config.config
        self.num_classes = None

    def get_dataloader(self, df):
        """
        Gets a dataloader for DataFrame `df`
        ----
        Trains the model using 'train_df'
        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.
        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        # cjr
        verbose = True
        # /cjr

        #if args:
        #    self.args.update_from_dict(args)

        assert "text" in df.columns and "labels" in df.columns, "must have dataframe with 'text' and 'labels' columns"
        the_examples = [
            InputExample(i, text, None, label)
            for i, (text, label) in enumerate(zip(df["text"].astype(str), df["labels"]))
        ]
        the_dataset = self.load_and_cache_examples(the_examples, verbose=verbose)
        #train_sampler = RandomSampler(train_dataset)
        the_dataloader = DataLoader(
            the_dataset,
            #sampler=train_sampler,
            #batch_size=self.args.train_batch_size,
            #num_workers=self.args.dataloader_num_workers,
        )
        return the_dataloader

    def get_num_classes(self):
        if not self.num_classes:
            # Use train_df to determine number of labels
            train_df_filename = self.config.get_output_data_dir() + 'train_df.csv'
            train_df = pd.read_csv(train_df_filename)
            self.num_classes = len(train_df['labels'].unique())
        return self.num_classes

    def get_my_dataloaders(self):
        """Offer the DataFrames with the original data through a PyTorch dataloader."""
        # 'full_df' is the filename of 'train_df' + 'eval_df'
        train_df_filename = self.config.get_output_data_dir() + 'full_df.csv'
        train_df = pd.read_csv(train_df_filename)

        # As we now already have train_df, use it to define num_classes if not set yet
        if not self.num_classes:
            # Use train_df to determine number of labels
            self.num_classes = len(train_df['labels'].unique())

        print("unique len labels: ", len(train_df['labels'].unique()))

        test_df_filename = self.config.get_output_data_dir() + 'test_df.csv'
        test_df = pd.read_csv(test_df_filename)

        train_dataloader = self.get_dataloader(train_df)
        test_dataloader = self.get_dataloader(test_df)

        return train_dataloader, test_dataloader

    def execute(self):
        WITH_GPU = True

        ptif.init_logging()
        ptif_config = ptif.get_default_config()
        ptif_config['gpu'] = int(WITH_GPU)
        ptif_config['num_classes'] = self.get_num_classes()

        trainloader, testloader = self.get_my_dataloaders()

        # We need the PyTorch model, rather than the SimpleTransformers wrapper around it
        pytorch_model = self.model
        if WITH_GPU:
            pytorch_model = pytorch_model.cuda()

        # Contrary to documentation: this function returns a dictionary, not a 3-element-tuple
        influences = ptif.calc_item_wise(ptif_config, pytorch_model, trainloader, testloader)

        output_file = self.config.get_output_data_dir() + 'if.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(influences, f)


class CreatePrediction(object):
    def __init__(self, model_type, model_name, **kwargs):
        self.model_type = model_type
        self.model_name = model_name
        self.keyword_args = kwargs
        self.config = pipeline_config.config

    def execute(self):
        print("Executing 'CreatePrediction")
        print("using arguments: ", self.keyword_args)

        # Read a datafile
        dev_file = self.keyword_args['devfile']

        with open(dev_file, 'r') as f:
            data = json.load(f)

        lst = []
        for item in data:
            id = item['id']
            txt = item['text']
            if 'labels' in item.keys():
                labs = item['labels']
            else:
                labs = []
            lst.append((id, txt, labs))

        # And now do something with it
        #df = pd.DataFrame(lst)
        #df.columns = ['id', 'text']

        # Convert sentences to lowercase
        # FIXME: hardcoded now, but depends on training also taking place on lowercased input
        #sentences = [s.lower() for s in df['text'].to_list()]
        #sentences = df['text'].to_list()
        sentences = [d['text'] for d in data]
        #sentences = [d['text'].lower() for d in data]

        # Get the model
        model_class, model_args = self.config.get_model_and_args_class(context='postprocess')
        model = model_class(
                self.keyword_args['model']['model_type'],
                self.keyword_args['model']['model_path'],
                args=model_args
        )

        # Predict outcome for all items
        outcomes, raw_outcomes = self.prediction_wrapper(model, sentences)
        #outcomes, _ = model.predict(sentences)
        #print(outcomes)

        labels = get_labels()
        label_identifier = pipeline_config.config.pipeline_config['data']['labels']

        for i in range(len(lst)):
            predicted_labels = prediction_to_labels(
                labels,
                converter.raw_to_onehot(raw_outcomes[i], label_identifier=label_identifier)
            )
            #predicted_labels = prediction_to_labels(labels, outcomes[i])
            data[i]['labels'] = predicted_labels
            #del data[i]['text']
            origlabels = lst[i][2]
            #data[i]['origlabels'] =origlabels

            if 'image' in data[i].keys():
                del data[i]['image']

        # write outcome to file
        predict_file = self.keyword_args['outfile']
        p = predict_file.split('.')
        predict_file = "{}_{}.{}".format(''.join(p[:-1]), datetime.today().strftime('%Y-%m-%d-%H_%M_%S'), p[-1])
        print('Writing outcome to ', predict_file)
        with open(predict_file, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def prediction_wrapper(self, model, sentences):
        def plus(o1, o2, raw1, raw2):
            for i in range(len(o1)):
                if o2[i] > 0:
                    o1[i] = 1
                raw1[i] = raw1[i] + raw2[i]
            return o1, raw1

        outcome = []
        raw_output = []
        for s in sentences:
            # First create a prediction for the whole article
            o, raw = model.predict([s])

            res_outcome = [0] * len(o[0])  # create a list with zeros, the size of a multilabel list
            res_raw = [0.0] * len(o[0])

            res_outcome, res_raw = plus(res_outcome, o[0], res_raw, raw[0])

            subs = [x for x in s.split('\n') if x != '']
            o, raw = model.predict(subs)
            for j in range(len(o)):
                res_outcome, res_raw = plus(res_outcome, o[j], res_raw, raw[j])
            outcome.append(res_outcome)
            raw_output.append(res_raw)
        return outcome, raw_output


class CreateSpanPrediction(object):
    def __init__(self, model_type, model_name, **kwargs):
        self.model_type = model_type
        self.model_name = model_name
        self.keyword_args = kwargs
        self.config = pipeline_config.config

    def execute(self):
        print("Executing 'CreateSpanPrediction")
        print("Using arguments: ", self.keyword_args)

        all_labels = get_labels()

        # Read a datafile
        dev_file = self.keyword_args['devfile']

        with open(dev_file, 'r') as f:
            data = json.load(f)

        lst = []
        for item in data:
            id = item['id']
            txt = item['text']
            orig_labels = item['labels']
            lst.append((id, txt, orig_labels))

        # And now do something with it
        #df = pd.DataFrame(lst)
        #df.columns = ['id', 'text']

        #sentences = df['text'].to_list()
        #ids = df['id'].to_list()

        sentences = [x[1] for x in lst]
        ids = [x[0] for x in lst]
        orig_labels = [x[2] for x in lst]

        # Hardcoded
        # Don't use CUDA, as we need it for the Seq2Seq model
        multilabel_model = MultiLabelClassificationModel(
            model_type='roberta',
            model_name='../outputs/outputs_quality',
            use_cuda=False
        )

        # We want the raw output of the multi-label
        _, raw_output = multilabel_model.predict(sentences)

        all_labels = get_labels()

        # Initialize outputs
        outputs = []
        for id, input_s, orig_labels in lst:
            outputs.append({'id': id, 'text': input_s, 'orig_labels': orig_labels, 'labels': []})

        # Get the model
        model_class, model_class_args = self.config.get_model_and_args_class(context='postprocess')
        model_type = self.keyword_args['model']['model_type']
        model_path = self.keyword_args['model']['model_path']
        model_args = model_class_args()
        if 'args' in self.keyword_args['model']:
            model_args.update_from_dict(self.keyword_args['model']['args'])

        additional_special_tokens = []
        for l in all_labels:
            _in_sym_, _out_sym_ = label_to_symbol(l, all_labels)
            additional_special_tokens.append(_in_sym_)
            additional_special_tokens.append(_out_sym_)

        if model_type == 'bart':
            model = model_class(
                encoder_decoder_type=model_type,
                encoder_decoder_name=model_path,
                additional_special_tokens_encoder=dict(additional_special_tokens=additional_special_tokens),
                additional_special_tokens_decoder=dict(additional_special_tokens=additional_special_tokens),
                use_cuda=True,
                args=model_args
            )
        else:
            # FIXME: now hardcoded
            if model_path == 'roberta-base':
                decoder_name = 'bert-base-uncased'
                encoder_name = model_path
                model = model_class(
                    encoder_type=model_type,
                    encoder_name=encoder_name,
                    decoder_name=decoder_name,
                    use_cuda=False,
                    additional_special_tokens_encoder=dict(additional_special_tokens=additional_special_tokens),
                    additional_special_tokens_decoder=dict(additional_special_tokens=additional_special_tokens),
                    args=model_args
                )
            else:
                decoder_name = model_path + '/decoder'
                encoder_name = model_path + '/encoder'
                model = model_class(
                    encoder_type=model_type,
                    encoder_name=encoder_name,
                    decoder_name=decoder_name,
                    use_cuda=False,
                    args=model_args
                )

        # Predict outcome for all items
        out_sentences = model.predict(sentences)  #batched_predict(model, sentences)

        with open('fin.txt', 'w') as fin:
            fin.writelines([f'{str(i)} {s}\n' for i, s in enumerate(sentences)])

        with open('fout.txt', 'w') as fout:
            fout.writelines([f'{str(i)} {s}\n' for i, s in enumerate(out_sentences)])

        split_sentence_predictions = []

        x = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p','q', 'r']
        for s_index in range(len(sentences)):
            if SPLIT_IN_SENTENCES:
                dirtylabels = {}  # any label mentioned in the outcome
                sublist = split_in_sentences(sentences[s_index])
                suboutcomes = model.predict(sublist)
                split_sentence_predictions.append((sublist, suboutcomes))
                article = ''
                fragments = []
                joined_outcome = ""
                for i, sub in enumerate(suboutcomes):
                    if type(sub) == list:
                        print("SUB = ", sub)
                        sub = sub[0]
                        joined_outcome += '\n' + sub

                        for l in get_all_dirty_labels(sub):
                            dirtylabels.add(l)

                    sub_article, sub_fragments, _ = decode(sub, all_labels)

                    # BEGIN alternative
                    #tokenized_list = insert_tags_list(sublist[i], sub)
                    #tagged_sentence = insert_tags(sublist[i], tokenized_list)
                    #sub_article, sub_fragments, errors = decode(tagged_sentence, all_labels)
                    #print("SENTENCES=[",sublist[i],"]")
                    #print("OUT SENTENCE=[",sub,"]")
                    #print("TAGGED SENTENCE=[",tagged_sentence, "]")
                    # END alternative

                    for sf in sub_fragments:
                        fragments.append(sf + len(article))  # Use __add__ function of Fragment
                    article += sub_article
                print("JOINED outcome = ", joined_outcome)
                print("OUTCOME = ", out_sentences[s_index][0])
                outputs[s_index]['dirtylabels'] = list(dirtylabels)
            else:
                print("===== Taking whole SENTENCES")
                if type(out_sentences[0]) == list:
                    print("====== getting returns as list")
                    top_article = None
                    top_fragments = None
                    min_errors = 1000
                    for i in range(len(out_sentences[0])):
                        # original
                        article, fragments, errors = decode(out_sentences[s_index][i], all_labels)

                        orig_text = data[s_index]['text']

                        # BEGIN alternative
                        #lst = insert_tags_list(orig_text, out_sentences[s_index][i])
                        #tagged_sentence = insert_tags(orig_text, lst)
                        #article, fragments, errors = decode(tagged_sentence, all_labels)
                        # END alternative

                        print(f"orig ({len(orig_text)}: {orig_text}")
                        for j, f in enumerate(fragments):
                            print(f"{x[j]}: {f.extract(orig_text)}")
                        print(f"generated ({len(article)}): {article}")
                        for j, f in enumerate(fragments):
                            print(f"{x[j]}: [{f.label}] {f.extract(article)}")
                        id = data[s_index]['id']
                        orig_labels = data[s_index]['labels']
                        print("original labels: ", orig_labels)

                        if errors < min_errors:
                            top_article = article
                            top_fragments = fragments
                            min_errors = errors
                    article = top_article
                    fragments = top_fragments
                else:
                    #tokenized_list = insert_tags_list(sentences[s_index], out_sentences[s_index])
                    #tagged_sentence = insert_tags(sentences[s_index], tokenized_list)
                    #article, fragments, errors = decode(tagged_sentence, all_labels)

                    #print("SENTENCES=[",sentences[s_index],"]")
                    #print("OUT SENTENCE=[",out_sentences[s_index],"]")
                    #print("TAGGED SENTENCE=[",tagged_sentence, "]")

                    article, fragments, errors = decode(out_sentences[s_index], all_labels)

            labels = []
            for i in range(len(fragments)):
                fragments[i] = calibrate(fragments[i], article, sentences[s_index], distance=8)

            for f in fragments:
                labels.append({'start': f.start, 'end': f.end,
                               'technique': f.label,
                               'text_fragment': f.extract(sentences[s_index])})
            outputs[s_index]['labels'] = labels

        #for sub_in, sub_out in split_sentence_predictions:
        #    print('sub_in', sub_in, 'sub_out', sub_out)

        # The SpanPrediction function can be used to merely predict labels.
        # If the name of the prediction file contains the term 'task1', the output
        # format will conform to Task1, rather than Task2
        if dev_file.find('task1') > 0:
            # We are matching task1 instead of task2. We only want labels now
            for i in range(len(outputs)):
                outputs[i]['labels'] = list({l['technique'] for l in outputs[i]['labels']})

        # write outcome to file
        predict_file = self.keyword_args['outfile']
        p = predict_file.split('.')
        # FIXME: for now not timestamping output file
        #predict_file = "{}_{}.{}".format(''.join(p[:-1]), datetime.today().strftime('%Y-%m-%d-%H_%M_%S'), p[-1])

        print('Writing outcome to ', predict_file)
        with open(predict_file, 'w', encoding='utf8') as f:
            json.dump(outputs, f, indent=4, ensure_ascii=False)


    def predict(self, cur_label, all_labels, sequence_model, multilabel_model, article):
        l_index = all_labels.index(cur_label)

        # Split the article in sentences; filter out empty sentences (\n\n)
        sub_list = [x for x in article.split('\n') if x != '']

        # Ignore multi-sentence fragments now
        predicted_labels = []

        # First we get outcomes for multilabel prediction
        outcomes, raw = multilabel_model.predict(sub_list)
        for i in range(len(sub_list)):
            predicted_labels.append(
                prediction_to_labels(
                    all_labels,
                    self.converter.raw_to_onehot(raw[i], label_identifier='semeval2021')
                )
            )

        # Next we get predictions for sequence prediction
        sequence_list = sequence_model.predict(sub_list)

        # Now we work through each of the sentences
        hit = 0
        fragments = []
        for i in range(len(sub_list)):
            sentence = sub_list[i]
            labels = predicted_labels[i]
            raw_i = raw[i]
            seqs = sequence_list[i]

            confidence = raw_i * len(all_labels)

            if cur_label in predicted_labels or confidence > 0.5:
                f = fragment_from_text(sentence, seqs, cur_label)
                if f is not None and f[1] > 0:
                    hit += 1
                    fragments.append({'start': f[0], 'end': f[1], 'technique': f[2], 'text_fragment': f[3]})

if __name__ == '__main__':
    config = pipeline_config.config
    wrapper = InfluenceFunction(
        config.get_model_type(),
        config.get_output_dir()
    )
    wrapper.execute()
    print('Completed Influence Function')