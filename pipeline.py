from train import ClassificationTrainer, SequenceTrainer
from eval import SequenceEvaluator, ClassificationEvaluator, SpanEvaluator
import pipeline_config
from load_data import CSVLoader, DataLoader
from label_tool import read_labels_from_file
import sys
import os

logger = pipeline_config.logger


class Pipeline(object):
    def __init__(self):
        self.config = pipeline_config.config
        #self.pipeline_config = self.config.pipeline_config

    def get_data_loader(self):
        data_phase = self.pipeline_config['data']
        data_module = __import__('load_data')


        if 'csv' in data_phase.keys():
            # For reading 'manually' filtered data from a saved dataframe
            csv_filename = data_phase['csv']
            csv_file = self.config.get_output_data_dir() + csv_filename
            self.dataloader = CSVLoader()
            self.dataloader.set_csv_file(csv_file)
        else:
            data_func = getattr(data_module, data_phase['dataloader'])
            data_file = data_phase['file']
            data = data_func(data_file)
            self.dataloader = DataLoader(label_identifier=data_phase['labels'])
            self.dataloader.set_data(data)
            if 'filters' in data_phase.keys():
                # Every filter is a function name in load_data taking a data_loader and returning it filtered
                data_filters = data_phase['filters']
                for filter_function_name in data_filters:
                    filter_function_name = 'filter_' + filter_function_name
                    filter_func = getattr(data_module, filter_function_name)
                    self.dataloader = filter_func(self.dataloader)

        return self.dataloader


    def execute(self):
        def capitalize(s):
            return s[0].upper()+s[1:]

        scenario = self.config.get_scenario()
        print(f"******* starting pipeline {self.config['pipeline']['cur_scenario']} ******")
        for current_step in self.config.get_scenario():
            print(f" =================== Current step: {current_step} =============================")
            self.config.set_current_pipeline_step(current_step)
            self.pipeline_config = self.config.pipeline_config

            logger.info('******** LOADING DATA *******')
            self.get_data_loader()

            # Training phase
            logger.info('******* TRAINING *******')
            print("get task type =========== ", self.config.get_task_type())
            if not 'disable' in self.pipeline_config['train'].keys() or \
                    self.pipeline_config['train']['disable'] is not True:
                train_phase = self.pipeline_config['train']
                train_module = __import__('train')
                converter_module = __import__('converter')
                converter_class = getattr(converter_module, capitalize(train_phase['converter'])+'Converter')
                if self.config.get_task_type() == 'sequence':
                    trainer_class_name = 'SpanTrainer'
                else:
                    trainer_class_name = 'ClassificationTrainer'
                trainer_class = getattr(train_module, trainer_class_name)
                trainer = trainer_class(self.dataloader, converter_class())
                trainer.train()
            else:
                logger.info(">>> training disabled")
                train_phase = self.pipeline_config['train']
                train_module = __import__('train')
                converter_module = __import__('converter')
                converter_class = getattr(converter_module, capitalize(train_phase['converter'])+'Converter')

            # Evaluation phase
            logger.info('******* EVALUATION *******')
            if not 'disable' in self.pipeline_config['eval'].keys() or \
                    self.pipeline_config['eval']['disable'] is not True:

                if self.config.get_task_type() == 'sequence':
                    #evaluator = SequenceEvaluator(self.dataloader)
                    evaluator = SpanEvaluator(self.dataloader)
                else:
                    evaluator = ClassificationEvaluator(self.dataloader)

                evaluator.evaluate(converter_class())

                # Create a score file for evaluation
                if self.config.get_task_type() != 'sequence':
                    from score import score_task1
                    pred_file = self.config.get_output_data_dir() + 'predict1.txt'
                    gold_file = self.config.get_output_data_dir() + 'true1.txt'
                    score_task1(
                        predict_file=pred_file,
                        true_file=gold_file
                    )

                    # Run the scorer
                    sys.path.append(os.path.realpath('SEMEVAL-2021-task6-corpus'))

                    from scorer.task1_3 import evaluate, validate_files  # (pred_fpath, gold_fpath, CLASSES):
                    # from format_checker.task1_3 import validate_files
                    CLASSES = read_labels_from_file(self.pipeline_config['data']['labels'])

                    if validate_files(pred_file, gold_file, CLASSES):
                        logger.info('Prediction file format is correct')
                        macro_f1, micro_f1 = evaluate(pred_file, gold_file, CLASSES)
                        logger.info("macro-F1={:.5f}\tmicro-F1={:.5f}".format(macro_f1, micro_f1))
                    else:
                        print("Failed to validate prediction & gold files")

                else:
                    print("No scoring for sequence type")

            else:
                print("Evaluation is disabled")

            # Post-processing phase
            if 'postprocess' in self.pipeline_config.keys() and \
                    self.pipeline_config['postprocess']['disable'] is not True:
                logger.info('******* POST-PROCESSING *******')
                postprocess_phase = self.pipeline_config['postprocess']
                postprocess_module = __import__('postprocess')
                postprocess_class = getattr(postprocess_module, postprocess_phase['processor']['class'])
                postprocessor = postprocess_class(
                    self.config.get_model_type(),
                    self.config.get_output_dir(),
                    **postprocess_phase['processor']  # This should return a dictionary
                )
                postprocessor.execute()

            else:
                print('No post-processing defined')



if __name__ == '__main__':
    print(pipeline_config.config)
    pipe = Pipeline()
    pipe.execute()
    print('done')





