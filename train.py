""""Train a model based on various parts of the dataset.
"""

from load_data import (
    SpanFilter,
    DataLoader,
    DataLoader2020_task2,
    DataLoader2021_task2
)
from converter import Converter
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
from abc import abstractmethod
import pipeline_config
import time
import torch

from label_tool import label_to_symbol, symbol_to_label

logger = pipeline_config.logger

class Trainer(object):
    def __init__(self, dataloader: DataLoader, converter: Converter, random_state=42):
        self.config = pipeline_config.config
        self.pipeline_config = self.config.pipeline_config
        self.dataloader = dataloader
        self.converter = converter
        self.random_state = random_state
        self._prepare_data()
        self._prepare_model()


    @abstractmethod
    def train(self):
        pass


    @abstractmethod
    def read_data(self):
        pass

    @abstractmethod
    def _prepare_data(self):
        pass

    @abstractmethod
    def _prepare_model(self):
        pass

    @abstractmethod
    def create_dataset(self):
        """Override to return X and y."""
        return [], []

    def get_format(self):
        cn = self.__class__.__name__
        ci_lst = [cn[0]]
        for i in range(1, len(cn)):
            if 'a' <= cn[i] <= 'z':
                ci_lst.append(cn[i])
            else:
                break
        return ''.join(ci_lst).lower()



class ClassificationTrainer(Trainer):
    def __init__(self, dataloader: DataLoader, converter: Converter, random_state=42):
        super().__init__(dataloader, converter, random_state=random_state)

    def _prepare_model(self):
        # Select the class for modeling based on configuration
        model_class, model_class_args = self.config.get_model_and_args_class()
        # print(model_class, model_class_args)

        # Optional model configuration
        model_args = model_class_args()
        model_args.output_dir = self.config.get_output_dir()
        if 'args' in self.config.get_model_object():
            model_args.update_from_dict(self.config.get_model_args())

        # print('model args = ', model_args)

        model_path_or_name = self.config.get_model_name()

        self.model = model_class(
            self.config.get_model_type(),
            self.config.get_model_name_or_path(),
            num_labels=self.num_labels,  # Note that num_labels doesn't go into 'args'
            args=model_args
        )

    def _prepare_data(self):
        self.labels, self.data = self.dataloader.load()
        len_articles = len(self.data)
        len_fragments = sum([len(a['fragments']) for a in self.data])
        len_labels = len(self.labels)
        self.num_labels = len_labels
        logger.info(f'Loaded data set: {len_articles} articles, {len_fragments} fragments,  {len_labels} categories')

        logger.info(f'Converter: {str(self.converter)}')
        X, y = self.converter(self.dataloader)
        logger.info(f"Sample input to the model: {X[0]} -> {y[0]}")
        skip_test = self.config.pipeline_config['train']['skip_test']
        X_train, y_train, X_eval, y_eval, X_test, y_test = self.converter.split(X, y, skip_test=skip_test)

        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.X_test = X_test
        self.y_test = y_test

    def read_data(self):
        train_df = self.converter.get_dataframe_for_dataset(self.X_train, self.y_train)
        eval_df = self.converter.get_dataframe_for_dataset(self.X_eval, self.y_eval)
        test_df = self.converter.get_dataframe_for_dataset(self.X_test, self.y_test)

        # FIXME: move this functionality to a more elegant place
        # Save test data to CSV file for later usage in evaluation
        train_df_filename = self.config.get_output_data_dir() + 'train_df.csv'
        train_df.to_csv(train_df_filename)

        eval_df_filename = self.config.get_output_data_dir() + 'eval_df.csv'
        eval_df.to_csv(eval_df_filename)

        # Duplicate, but convenient for postprocessing to have a version with a single index
        full_df_filename = self.config.get_output_data_dir() + 'full_df.csv'
        full_df = train_df.append(eval_df, ignore_index=True)
        full_df.to_csv(full_df_filename)

        test_df_filename = self.config.get_output_data_dir() + 'test_df.csv'
        test_df.to_csv(test_df_filename)
        return train_df, eval_df

    def train(self):
        train_df, eval_df = self.read_data()
        self.model.train_model(train_df, eval_df=eval_df)


class SequenceTrainer(Trainer):
    def __init__(self, dataloader: DataLoader, converter: Converter, random_state=42):
        super().__init__(dataloader, converter, random_state=random_state)

    def _get_model_for_label(self, label, use_cuda=True, num_train_epochs=None):
        label_index = self.config.modelled_labels.index(label)
        model_class, model_class_args = self.config.get_model_and_args_class()
        model_args = model_class_args()
        if 'args' in self.config.get_model_object():
            model_args.update_from_dict(self.config.get_model_args())

        # Set a different output directory for each model, under the standard output directory
        model_args.output_dir = self.config.get_output_dir() + '/label_' + str(label_index)
        if num_train_epochs is not None:
            model_args.num_train_epochs = num_train_epochs
        model_name_or_path = self.config.get_model_name_or_path()
        # Either we have a huggingface model with 'bart' in the name
        # or we have a previously trained model specific for a label. If the latter, add label index to path
        if model_name_or_path.find('bart') < 0 and model_name_or_path.find('roberta') < 0:
            model_name_or_path += '/label_' + str(label_index)
        return model_class(
                encoder_decoder_type=self.config.get_model_type(),
                encoder_decoder_name=model_name_or_path,
                use_cuda=use_cuda,
                args=model_args
            )

    def _prepare_model(self):
        """Don't prepare a model, generate them on demand with _get_model_for_label"""
        pass

    def _prepare_data(self):
        """Create a dataset for each label that is to be modelled"""
        self.data_per_label = {}
        self.labels, self.data = self.dataloader.load()
        len_articles = len(self.data)
        len_fragments = sum([len(a['fragments']) for a in self.data])

        print(f'Loaded data set: {len_articles} articles, {len_fragments} fragments,  {len(self.labels)} categories')

        logger.info(f'Converter: {str(self.converter)}')
        X, y = self.converter(self.dataloader)

        # Create a dataframe with data for each label
        for index, l in enumerate(self.config.modelled_labels):
            full_df = self.converter.get_dataframe_for_dataset(X, y)
            # We set the target text to the empty string for all labels other than the chosen one
            full_df.loc[full_df['label'] != l, 'target_text'] = ''

            # Remove all target texts that are too long

            logger.info(f">>> Loading data for label *{l}*")
            import pprint
            pprint.pprint(full_df.head(20))

            # Split into train, evaluation, and test sets
            rest_df, test_df = train_test_split(full_df, test_size=0.1, random_state=self.random_state)
            train_df, eval_df = train_test_split(rest_df, test_size=0.2, random_state=self.random_state)

            # Save these dataframes for later use
            train_df_filename = self.config.get_output_data_dir() + str(index) + '_train_df.csv'
            train_df.to_csv(train_df_filename)

            eval_df_filename = self.config.get_output_data_dir() + str(index) + '_eval_df.csv'
            eval_df.to_csv(eval_df_filename)

            test_df_filename = self.config.get_output_data_dir() + str(index) + '_test_df.csv'
            test_df.to_csv(test_df_filename)

            # Duplicate, but convenient for postprocessing to have a version with a single index
            full_df_filename = self.config.get_output_data_dir() + str(index) + '_full_df.csv'
            full_df.to_csv(full_df_filename)

            self.data_per_label[l] = (train_df, eval_df, test_df)

    def train(self):
        for label in self.config.modelled_labels:
            train_df, eval_df, _ = self.data_per_label[label]
            # Don't train if there are no target texts defined in the training data
            if (len(train_df[train_df['target_text'] != '']) > 0):
                # Don't train for empty target texts (for efficiency's sake),
                # but leave the empty targets in the evaluation data
                # uncomment below to NOT train for empty targets
                # commented means that empty targets will also be trained.
                model = self._get_model_for_label(label, use_cuda=True)
                train_df = train_df[train_df['target_text'] != '']
                logger.info(f">>> training sequence model for label *{label}* ({len(train_df)} items)")
            else:
                # Fake training on first (empty) item, reason: create directory and copy model
                # for evaluation and for next step in pipeline
                model = self._get_model_for_label(label, use_cuda=True, num_train_epochs=1)
                train_df = train_df.head(1)
                logger.info(f"<<< No target texts for label *{label}*, skipping training (except for fake step)")

            try:
                model.train_model(train_df, eval_data=eval_df)
            except RuntimeError:
                logger.info("Falling back on NON-CUDA model")
                model = self._get_model_for_label(label, use_cuda=False)
                model.train_model(train_df, eval_data=eval_df)

            del model

            timeout = 10
            print(f"+++ Sleeping for {timeout} seconds to release CUDA")
            time.sleep(int(timeout/2))
            #torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            time.sleep(int(timeout/2))

    def read_data(self):
        pass

    def _read_data_FIXME_REMOVE(self):
        if self.dataloader.ready_made_data:
            full_df = self.dataloader.get_dataframe()
            train_df, eval_df = train_test_split(full_df, test_size=0.2, random_state=self.random_state)

        else:
            train_df = self.converter.get_dataframe_for_dataset(self.X_train, self.y_train)
            eval_df = self.converter.get_dataframe_for_dataset(self.X_eval, self.y_eval)
            test_df = self.converter.get_dataframe_for_dataset(self.X_test, self.y_test)

            # FIXME: move this functionality to a more elegant place
            # Save test data to CSV file for later usage in evaluation
            train_df_filename = self.config.get_output_data_dir() + 'train_df.csv'
            train_df.to_csv(train_df_filename)

            eval_df_filename = self.config.get_output_data_dir() + 'eval_df.csv'
            eval_df.to_csv(eval_df_filename)

            # Duplicate, but convenient for postprocessing to have a version with a single index
            full_df_filename = self.config.get_output_data_dir() + 'full_df.csv'
            full_df = train_df.append(eval_df, ignore_index=True)
            full_df.to_csv(full_df_filename)

            test_df_filename = self.config.get_output_data_dir() + 'test_df.csv'
            test_df.to_csv(test_df_filename)
        return full_df


class SpanTrainer(Trainer):
    def __init__(self, dataloader: DataLoader, converter: Converter, random_state=42):
        super().__init__(dataloader, converter, random_state=random_state)

    def _get_model(self, use_cuda=True, num_train_epochs=None):
        model_class, model_class_args = self.config.get_model_and_args_class()
        model_args = model_class_args()
        if 'args' in self.config.get_model_object():
            model_args.update_from_dict(self.config.get_model_args())

        # Set a different output directory for each model, under the standard output directory
        model_args.output_dir = self.config.get_output_dir() #  + '/label_' + str(label_index)
        if num_train_epochs is not None:
            model_args.num_train_epochs = num_train_epochs
        model_name_or_path = self.config.get_model_name_or_path()
        # Either we have a huggingface model with 'bart' in the name
        # or we have a previously trained model specific for a label. If the latter, add label index to path
        #if model_name_or_path.find('bart') < 0 and model_name_or_path.find('roberta') < 0:
        #    model_name_or_path += '/label_' + str(label_index)

        # NOTE: we can't add special tokens to Huggingface Encoder/Decoder models - like BART.
        # See: https://github.com/allenai/allennlp/pull/4946
        additional_special_tokens = []
        for l in self.labels:
            _in_sym_, _out_sym_ = label_to_symbol(l, self.dataloader.labels)
            additional_special_tokens.append(_in_sym_)
            additional_special_tokens.append(_out_sym_)
        if self.config.get_model_type() == 'bart':
            model = model_class(
                encoder_decoder_type=self.config.get_model_type(),
                encoder_decoder_name=model_name_or_path,
                use_cuda=use_cuda,
                additional_special_tokens_encoder=dict(additional_special_tokens=additional_special_tokens),
                additional_special_tokens_decoder=dict(additional_special_tokens=additional_special_tokens),
                args=model_args
            )
        else:
            # FIXME: now hardcoded
            if model_name_or_path == 'roberta-base':
                decoder_name = 'bert-base-uncased'
                encoder_name = model_name_or_path
                model = model_class(
                    encoder_type=self.config.get_model_type(),
                    encoder_name=encoder_name,
                    decoder_name=decoder_name,
                    use_cuda=use_cuda,
                    additional_special_tokens_encoder=dict(additional_special_tokens=additional_special_tokens),
                    additional_special_tokens_decoder=dict(additional_special_tokens=additional_special_tokens),
                    args=model_args
                )
            else:
                decoder_name = model_name_or_path + '/decoder'
                encoder_name = model_name_or_path + '/encoder'
                model = model_class(
                    encoder_type=self.config.get_model_type(),
                    encoder_name=encoder_name,
                    decoder_name=decoder_name,
                    additional_special_tokens_encoder=dict(additional_special_tokens=additional_special_tokens),
                    additional_special_tokens_decoder=dict(additional_special_tokens=additional_special_tokens),
                    use_cuda=use_cuda,
                    args=model_args
                )
        return model

    def _prepare_model(self):
        """Don't prepare a model, generate them on demand with _get_model_for_label"""
        pass

    def _prepare_data(self):
        """Create a dataset for each label that is to be modelled"""
        self.data_per_label = {}
        self.labels, self.data = self.dataloader.load()
        len_articles = len(self.data)
        len_fragments = sum([len(a['fragments']) for a in self.data])

        print(f'Loaded data set: {len_articles} articles, {len_fragments} fragments,  {len(self.labels)} categories')

        logger.info(f'Converter: {str(self.converter)}')
        X, y = self.converter(self.dataloader)

        # Create a dataframe with data for each label
        #for index, l in enumerate(self.config.modelled_labels):
        full_df = self.converter.get_dataframe_for_dataset(X, y)

        import pprint
        pprint.pprint(full_df.head(10))

        # Split into train, evaluation, and test sets
        rest_df, test_df = train_test_split(full_df, test_size=0.1, random_state=self.random_state)
        train_df, eval_df = train_test_split(rest_df, test_size=0.2, random_state=self.random_state)

        # Save these dataframes for later use
        train_df_filename = self.config.get_output_data_dir() + '/train_df.csv'
        train_df.to_csv(train_df_filename)

        eval_df_filename = self.config.get_output_data_dir() + '/eval_df.csv'
        eval_df.to_csv(eval_df_filename)

        test_df_filename = self.config.get_output_data_dir() + '/test_df.csv'
        test_df.to_csv(test_df_filename)

        # Duplicate, but convenient for postprocessing to have a version with a single index
        full_df_filename = self.config.get_output_data_dir() + '/full_df.csv'
        full_df.to_csv(full_df_filename)

        return train_df, eval_df, test_df

    def train(self):
        train_df, eval_df, _ = self._prepare_data()
        # Don't train if there are no target texts defined in the training data
        model = self._get_model(use_cuda=True)

        try:
            model.train_model(train_df, eval_data=eval_df)
        except RuntimeError:
            logger.info("Falling back on NON-CUDA model")
            model = self._get_model(use_cuda=False)
            model.train_model(train_df, eval_data=eval_df)

    def read_data(self):
        pass



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    dataloader = DataLoader2020_task2()

    print('output dir = ', trainer.config.get_output_dir())
    trainer.train()

    print("Training finished. Use eval.py to evaluate the outcome.")