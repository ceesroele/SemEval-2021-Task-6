"""Use different techniques to evaluate outcome of training"""

import os
import pickle
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from load_data import DataLoader, DataLoader2020_task2
import pandas as pd
import numpy as np
import logging
import pipeline_config
from converter import Converter
from label_tool import label_to_symbol, symbol_to_label
from sklearn.metrics import f1_score


from sklearn.preprocessing import MultiLabelBinarizer

logger = pipeline_config.logger

def batched_predict(model, lst):
    """Wrap the model.predict(..) method in a function that makes calls using subsets.

    Reason: when calling model.predict(..) on a full list I'm liable to get Out-Of-Memory exceptions.
    """
    res = []
    batch_size = 8
    start = 0
    while start < len(lst):
        res.extend(model.predict(lst[start:start+batch_size]))
        start += batch_size
    assert len(lst) == len(res)
    return res

def get_model():
    config = pipeline_config.config
    model_class, _ = config.get_model_and_args_class()
    print("EVUATION, getting model type", config.get_model_type(), "from directory", config.get_model_dir(context='eval'))
    model = model_class(
        config.get_model_type(),
        config.get_model_dir(context='eval'),
        args={'reprocess_input_data': True}
    )

class Evaluator(object):
    def __init__(self, dataloader: DataLoader):
        self.config = pipeline_config.config
        self.pipeline_config = self.config.pipeline_config
        self.dataloader = dataloader

    def wrong_predictions_to_df(self, wrong_predictions):
        lst = [(x.text_a, x.label) for x in wrong_predictions]
        wrong_df = pd.DataFrame(lst)
        wrong_df.columns = ['sentence', 'wrong_label']
        # Save for operations in Jupyter notebook
        wrong_df_filename = self.config.get_output_data_dir() + "wrong_df.csv"
        wrong_df.to_csv(wrong_df_filename)
        return wrong_df


class ClassificationEvaluator(Evaluator):
    def __init__(self, dataloader: DataLoader, model_path=None):
        super().__init__(dataloader)

        model_class, _ = self.config.get_model_and_args_class()
        if model_path is None:
            model_path = self.config.get_model_dir(context='eval')
        print("EVALUATION, getting model type", self.config.get_model_type(), "from directory", model_path)
        self.model = model_class(
                self.config.get_model_type(),
                model_path,
                args={'reprocess_input_data': True}
                )

    def predict(self, lst: list):
        predictions = []
        predictions, _ = self.model.predict(lst)
        if predictions and type(predictions[0]) == list:
            # multiple labels
            for p, s in zip(predictions, lst):
                labels = [self.dataloader.int_to_label(y) for y in range(len(p)) \
                      if p[y] == 1]
                print(s, ' --> ', labels, p)
        else:
            labels = [self.dataloader.int_to_label(x) for x in predictions]
            for l, p, s in zip(labels, predictions, lst):
                print(s, ' --> ', l)

    def evaluate(self, converter: Converter):
        testdf_filename = self.config.get_output_data_dir() + 'test_df.csv'
        # Remove empty values from test_df (FIXME: how did they get there?)
        test_df = pd.read_csv(testdf_filename)[['id', 'text', 'labels']].dropna()
        #print('len of test_df = ', len(test_df))
        #print('len of test_df.dropna() = ', len(test_df.dropna()))
        # Multi value labels are read from CSV as string, e.g. '[0, 1, 0]' instead of list([0,1,0]).
        # We need to manually convert that to a list.
        test_df['labels'] = test_df['labels'].apply(lambda x: eval(x) if type(x) == str else x)

        def f1(predictions, outputs):
            mlb = MultiLabelBinarizer()

            # test_df as a global
            return f1_score(
                mlb.fit_transform(test_df['labels']),
                mlb.fit_transform(predictions),
                average='micro'
            )

        gold = test_df['labels'].to_list()

        print('----- test_df --------')
        result, model_outputs, wrong_predictions = self.model.eval_model(test_df, f1=f1)
        #print('result = ', result)
        #print('model outputs = ', model_outputs)

        if self.pipeline_config['train']['target'] != 'multi-label classification':
            # There are no 'wrong predictions' defined for multi-label classification
            #print('wrong prediction: ', wrong_predictions[0])
            all = test_df.shape[0]
            wrong = len(wrong_predictions)
            #print('accuracy = ', ((all - wrong) / float(all)) * 100.0, '%')
            self.wrong_predictions_to_df(wrong_predictions)

        label_identifier = pipeline_config.config.pipeline_config['data']['labels']
        predictions = [(converter.raw_to_onehot(pred, label_identifier=label_identifier),)
                       for pred in model_outputs]
        #pred_df = pd.DataFrame(zip(predictions, gold, model_outputs.tolist())).rename(
        #    columns={0: 'predictions', 1: 'gold', 2: 'outputs'})
        pred_df = pd.DataFrame(predictions).rename(columns={0: 'predictions'})
        pred_df_filename = self.config.get_output_data_dir() + 'pred_outputs.csv'
        pred_df.to_csv(pred_df_filename)
        output_df = pd.concat([test_df, pred_df], axis=1)
        output_df_filename = self.config.get_output_data_dir() + 'eval_outputs.csv'
        output_df.to_csv(output_df_filename)

        # Save the model outputs
        # output_filename = 'outputs_'+config_identifier(self.format, self.model_name)+"/eval_model_outputs.pkl"
        # with open(output_filename, 'wb') as f:
        #    pickle.dump(model_outputs, f)


class SequenceEvaluator(Evaluator):
    def __init__(self, dataloader: DataLoader):
        super().__init__(dataloader)
        self.model_dict = {}
        for index, l in enumerate(self.config.modelled_labels):
            model_class, _ = self.config.get_model_and_args_class()
            model_path = self.config.get_model_dir(context='eval')
            model_path += '/label_' + str(index)
            self.model_dict[l] = model_class(
                encoder_decoder_type=self.config.get_model_type(),
                encoder_decoder_name=model_path,
                use_cuda=False
            )

    def predict(self, lst: list):
        # Predict per label
        for index, l in enumerate(self.config.modelled_labels):
            model = self.model_dict[l]
            print(f">>> predictions for label *{l}*")
            predictions = batched_predict(model, lst)
            for input_text, target_text in zip(lst, predictions):
                print('input[', input_text, '] => output[', target_text, ']')

    def evaluate(self, converter: Converter):
        # Test per label
        for index, l in enumerate(self.config.modelled_labels):
            model = self.model_dict[l]
            print(f">>> evaluation for label *{l}*")
            testdf_filename = self.config.get_output_data_dir() + str(index) + '_test_df.csv'
            # Remove empty values from test_df (FIXME: how did they get there?)
            test_df = pd.read_csv(testdf_filename)[['id', 'input_text', 'target_text']].dropna()
            #print('len of test_df = ', len(test_df))
            #print('len of test_df.dropna() = ', len(test_df.dropna()))
            # Multi value labels are read from CSV as string, e.g. '[0, 1, 0]' instead of list([0,1,0]).
            # We need to manually convert that to a list.
            # test_df['labels'] = test_df['labels'].apply(lambda x: eval(x) if type(x) == str else x)
            # results = self.model.eval_model(test_df)
            input_text_list = test_df['input_text'].to_list()
            target_text_list = test_df['target_text'].to_list()
            predictions = batched_predict(model, input_text_list)
            for prediction, target_text in zip(predictions, target_text_list):
                print('target[', target_text, '] ---> prediction[', prediction, ']')


class SpanEvaluator(Evaluator):
    def __init__(self, dataloader: DataLoader):
        super().__init__(dataloader)
        self.model_dict = {}
        model_class, _ = self.config.get_model_and_args_class()
        model_type = self.config.get_model_type()
        model_path = self.config.get_model_dir(context='eval')
        additional_special_tokens = []
        for l in dataloader.labels:
            _in_sym_, _out_sym_ = label_to_symbol(l, self.dataloader.labels)
            additional_special_tokens.append(_in_sym_)
            additional_special_tokens.append(_out_sym_)

        if model_type == 'bart':
            self.model = model_class(
                encoder_decoder_type=self.config.get_model_type(),
                encoder_decoder_name=model_path,
                additional_special_tokens_encoder=dict(additional_special_tokens=additional_special_tokens),
                additional_special_tokens_decoder=dict(additional_special_tokens=additional_special_tokens),
                use_cuda=False
            )
        else:
            # FIXME: now hardcoded
            if model_path == 'roberta-base':
                decoder_name = 'bert-base-uncased'
                encoder_name = model_path
                self.model = model_class(
                    encoder_type=self.config.get_model_type(),
                    encoder_name=encoder_name,
                    decoder_name=decoder_name,
                    additional_special_tokens_encoder=dict(additional_special_tokens=additional_special_tokens),
                    additional_special_tokens_decoder=dict(additional_special_tokens=additional_special_tokens),
                    use_cuda=False
                )
            else:
                decoder_name = model_path + '/decoder'
                encoder_name = model_path + '/encoder'
                self.model = model_class(
                    encoder_type=self.config.get_model_type(),
                    encoder_name=encoder_name,
                    decoder_name=decoder_name,
                    use_cuda=False
                )

    def predict(self, lst: list):
        predictions = batched_predict(self.model, lst)
        for input_text, target_text in zip(lst, predictions):
            print('input[', input_text, '] => output[', target_text, ']')

    def evaluate(self, converter: Converter):
        testdf_filename = self.config.get_output_data_dir() + 'test_df.csv'
        # Remove empty values from test_df (FIXME: how did they get there?)
        test_df = pd.read_csv(testdf_filename)[['id', 'input_text', 'target_text']].dropna()
        # Multi value labels are read from CSV as string, e.g. '[0, 1, 0]' instead of list([0,1,0]).
        # We need to manually convert that to a list.
        # test_df['labels'] = test_df['labels'].apply(lambda x: eval(x) if type(x) == str else x)
        # results = self.model.eval_model(test_df)
        input_text_list = test_df['input_text'].to_list()
        target_text_list = test_df['target_text'].to_list()
        predictions = batched_predict(self.model, input_text_list)
        for prediction, target_text in zip(predictions, target_text_list):
            print('target[', target_text, '] ---> prediction[', prediction, ']')

