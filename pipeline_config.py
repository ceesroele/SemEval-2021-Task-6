"""Read configuration file and make it available
"""

# Installed module: PyYAML
import yaml
import copy
import os
from simpletransformers.classification import (
    ClassificationModel,
    ClassificationArgs,
    MultiLabelClassificationModel,
    MultiLabelClassificationArgs,
)
from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)
from custom import CustomMultiLabelClassificationModel
import logging
import sys
from label_tool import read_labels_from_file

# Use this logger in all modules
logger = logging.getLogger("pipeline")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

CONFIG_FILE = 'dev.yaml'

def config_identifier(converter, model_name):
    """Create identifier of configuration based on data `converter` and `model_name`"""
    return model_name.lower().replace('-', '_') + '_' + converter


def get_output_dir(base_output_dir, converter, model_name):
    return base_output_dir + '/outputs_' + config_identifier(converter, model_name) + '/'


def cascade_overwrite_dict(default_dict, overwrite_dict):
    # FIXME: now only one level deep; make recursive
    merged = copy.deepcopy(default_dict)
    for k, v in overwrite_dict.items():
        if type(v) == dict:
            # recurse
            if k in merged:
                assert type(merged[k]) == dict, \
                    f"Can't overwrite non-dictionary {str(merged[k])} with dictionary {str(v)}"
                v2 = merged[k]
            else:
                v2 = {}
            for k_, v_ in v.items():
                v2[k_] = v_
            merged[k] = v2
        else:
            merged[k] = v
    return merged


class Config(object):
    def __init__(self):
        with open('dev.yaml', 'r') as f:
            self._config = yaml.load(f, Loader=yaml.SafeLoader)
        self.modelled_labels = read_labels_from_file("semeval2021")

    def get_scenario(self):
        cur_scenario = self._config['pipeline']['cur_scenario']
        return self._config['pipeline']['scenario'][cur_scenario]

    def set_current_pipeline_step(self, current_step):
        """Defines variables for current step in the pipeline, including the used model"""
        self.current_step = current_step
        current_pipeline = self._config['pipeline'][current_step]

        # Overwrite defaults with the current configuration
        self.pipeline_config = cascade_overwrite_dict(self._config['pipeline']['default'],
                                                      self._config['pipeline'][current_step])

        self.model_object = cascade_overwrite_dict(self._config['model']['default'],
                                                   self.pipeline_config['train']['model'])
        return self.pipeline_config

    def get_base_output_dir(self):
        self.pipeline_config['output_dir']

    def XXXget_model_reference(self):
        if 'model' in self.pipeline_config['train']:
            return self.pipeline_config['train']['model']
        else:
            return None

    def get_model_object(self):
        return self.model_object

    def get_model_name(self):
        if 'name' in self.model_object.keys() and self.model_object['name'] is not None:
            return self.model_object['name']
        else:
            return None

    def get_model_name_or_path(self):
        if 'name' in self.model_object.keys() and self.model_object['name'] is not None:
            return self.model_object['name']
        elif 'model_path' in self.model_object.keys():
            return self.model_object['model_path']
        else:
            raise RuntimeError("No model_name or model_path specified in configuration.")

    def get_model_type(self):
        print(self.model_object)
        return self.model_object['model_type']

    def get_task_type(self):
        """Classification or sequence"""
        con = self.get_converter()
        # include 'span' and 'span2'
        if con == 'span':
            return "sequence"
        elif con == 'span2':
            return "span"
        else:
            return 'classification'

    def get_model_args(self):
        d = self.model_object['args']
        # make sure floats are converted to objects
        d2 = {}
        for k, v in d.items():
            if type(v) is str:
                d2[k] = eval(v)
            else:
                d2[k] = v
        return d2

    def get_model_dir(self, context=None):
        """Directory in which the model can be found"""
        if context == 'eval' and \
            'eval' in self.pipeline_config.keys() and \
                'model_path' in self.pipeline_config['eval'].keys():
            model_path = self.pipeline_config['output_dir'] + '/' + self.pipeline_config['eval']['model_path']
            print(f'Using model path: {model_path}')
            return model_path
        elif context == 'train' and \
                'train' in self.pipeline_config.keys() and \
                'model_path' in self.pipeline_config['train'].keys() and \
                'model_type' in self.pipeline_config['train'].keys():
            return self.pipeline_config['train']['model_path']
        elif context == 'postprocess' and \
                    'postprocess' in self.pipeline_config.keys() and \
                    'model_path' in self.pipeline_config[context].keys():
                model_path = self.pipeline_config['output_dir'] + '/' + self.pipeline_config[context]['model_path']
                print(f'Using model path: {model_path}')
                return model_path
        else:
            # FIXME: now having a class method and also a function named get_output_dir, confusing!
            output_dir = self.get_output_dir()
            print(f'Getting model from {output_dir}')
            return output_dir

    def get_converter(self):
        return self.pipeline_config['train']['converter']

    def config_identifier(self):
        """Use the name of the currently selected pipeline element as identifier"""
        return self.current_step.lower().replace('-', '_')

    # FIXME: now having a class method and a function named get_output_dir, confusing!
    def get_output_dir(self):
        return self.pipeline_config['output_dir'] + '/outputs_' + self.config_identifier() + '/'

    def get_output_data_dir(self):
        dir = self.pipeline_config['output_dir'] + '/data_' + self.config_identifier() + '/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        return dir

    def get_model_and_args_class(self, context='train'):
        """Create a ClassificationModel
         Possible configuration arguments:
         pipeline:
           <name>:
              <context>:
                target: {classification, multi-label classification}
        """
        TASK_CLASSES = {
            'classification': (ClassificationModel, ClassificationArgs),
            'multi-label classification': (MultiLabelClassificationModel, MultiLabelClassificationArgs),
            'span identification': (Seq2SeqModel, Seq2SeqArgs),
            'span2 identification': (Seq2SeqModel, Seq2SeqArgs)
        }
        assert 'target' in self.pipeline_config[context], "No 'target' defined"
        target = self.pipeline_config[context]['target']
        assert target in TASK_CLASSES.keys(), f"Target model {target} not in {str(TASK_CLASSES)}"
        mc, mc_args = TASK_CLASSES[target]
        print(f'Modelling: {target} using class {mc}')
        return mc, mc_args

    def __getitem__(self, item):
        """Directly obtain items in self._config"""
        return self._config[item]

    def __str__(self):
        l = [f"Scenario: {self._config['pipeline']['cur_scenario']}: {self.get_scenario()}"]
             #f"Model reference: {self.get_model_reference()}",
             #f"Model type/name_or_path: {self.get_model_type()}/{self.get_model_name_or_path()}",
             #f"Model - args {self.model_object['args']}", f"Converter: {self.get_converter()}",
             #f"Output directory: {self.get_output_dir()}",
             #f"Data directory: {self.get_output_data_dir()}"]
        return '\n'.join(l)


# Define the configuration right into the class
config = Config()

if __name__ == '__main__':
    print(config)
