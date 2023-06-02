import logging
import os
import socket
from datetime import datetime
from functools import reduce, partial
from operator import getitem

from logger import setup_logging
from utils import *


# from pathlib import Path

def decide_paths():
    HOST = socket.gethostname()

    if HOST.endswith("cloudlab.us"):
        is_kaggle = False
        HOST = "cloudlab"
    kaggle_run_type = os.getenv("KAGGLE_KERNEL_RUN_TYPE")
    if kaggle_run_type is None:
        is_kaggle = False
    else:
        is_kaggle = True
        HOST = "kaggle"
        print("Kaggle run type: {}".format(kaggle_run_type))

    is_test = False
    is_train = True

    is_to_submit = kaggle_run_type == "Batch"

    if HOST == "cloudlab":
        ROOT_DIR = Path("/local/Codes/Vesuvius").absolute()
        DATA_DIR = ROOT_DIR / "data" / "raw"
        OUTPUT_DIR = ROOT_DIR / "saved"

        CP_DIR = OUTPUT_DIR / "checkpoints"
        LOG_DIR = OUTPUT_DIR / "logs"
        CACHE_DIR = OUTPUT_DIR / "cache"
        EXTERNAL_MODELS_DIR = ROOT_DIR / "model"

    elif HOST == "kaggle":
        ROOT_DIR = Path("/kaggle")
        DATA_DIR = ROOT_DIR / "input" / "vesuvius-challenge-ink-detection"
        OUTPUT_DIR = ROOT_DIR / "working" / "saved"

        CP_DIR = OUTPUT_DIR / "checkpoints"
        LOG_DIR = OUTPUT_DIR / "logs"
        CACHE_DIR = OUTPUT_DIR / "cache"
        EXTERNAL_MODELS_DIR = ROOT_DIR / "input"
    else:
        ROOT_DIR = Path("../../").absolute()
        DATA_DIR = ROOT_DIR / "data" / "raw"
        OUTPUT_DIR = ROOT_DIR / "saved"
        #
        CP_DIR = OUTPUT_DIR / "checkpoints"
        LOG_DIR = OUTPUT_DIR / "logs"
        CACHE_DIR = OUTPUT_DIR / "cache"
        EXTERNAL_MODELS_DIR = ROOT_DIR / "model"

    print(f"ROOT_DIR: {ROOT_DIR}")
    assert os.listdir(DATA_DIR) != [], "Data directory is empty"

    for p in [ROOT_DIR, DATA_DIR, OUTPUT_DIR, CP_DIR, LOG_DIR, CACHE_DIR]:
        if os.path.exists(p) is False:
            os.makedirs(p)

    return {
        "ROOT_DIR": ROOT_DIR,
        "DATA_DIR": DATA_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        # "CP_DIR": CP_DIR,
        # "LOG_DIR": LOG_DIR,
        # "CACHE_DIR": CACHE_DIR,
        "EXTERNAL_MODELS_DIR": EXTERNAL_MODELS_DIR,
        "HOST": HOST,
    }


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        PATHS = decide_paths()
        # set save_dir where trained model and log will be saved.
        save_dir = PATHS["OUTPUT_DIR"]
        self._data_dir = PATHS["DATA_DIR"]
        self._root_dir = PATHS["ROOT_DIR"]
        self.HOST = PATHS["HOST"]

        # Write a memo of exp_name, exp_id, and run_id
        Model_Proto = self.config["arch"]["Proto"]
        Model_type = self.config["arch"]["model_type"]
        Channel = self.config["arch"]["channel"]
        exp_name = f"{Model_Proto}_{Model_type}model_{Channel}chs"

        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / "checkpoints" / exp_name / run_id
        self._log_dir = save_dir / 'logs' / exp_name / run_id
        self._cache_dir = save_dir / 'cache'
        self._external_models_dir = PATHS["EXTERNAL_MODELS_DIR"]


        # add run_id to configs
        self.config['run_id'] = run_id
        # modification={'run_id': run_id}
        # self._config = _update_config(config, modification)

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_yaml(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=None):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if options is None:
            options = []
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        # config = read_json(cfg_fname)
        config = read_yaml(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            # config.update(read_json(args.config))
            config.update(read_yaml(args.config))
        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def init_obj(self, attr_name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[attr_name]['type']
        module_args = dict(self[attr_name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def __contains__(self, name):
        """Use in operation like ordinary dict."""
        return name in self.config

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
