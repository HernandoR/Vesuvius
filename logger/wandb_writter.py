import importlib


class WandbWriter():
    def __init__(self, log_dir, logger, enabled):
        self.wandb = None
        if enabled:
            succeeded = False
            for module in ["wandb"]:
                try:
                    self.wandb = importlib.import_module(module)

                    # self.wandb.login()

                    succeeded = True
                    logger.info("wandb is installed")
                    break
                except ImportError:
                    succeeded = False

        if not succeeded:
            msg = " warning: wandb is configured to use, but currently not installed on " \
                  "this machine. Please install wandb with 'pip install wandb' " \
                  "or turn off the option in the 'config.json' file."
            logger.warning(msg)
        else:
            self.wandb.tensorboard.patch(root_logdir=str(log_dir),
                                         pytorch=True)
            # self.wandb.init()
        self.step = 0
        self.mode = ''

        self.wandb_data_types = {'Audio', 'BoundingBoxed2D', 'Graph',
                                 'Histogram', 'HTML', 'Image', 'ImageMask',
                                 'Molecule', 'Object3D', 'Plotly', 'Table', 'Video'}
        # self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If wandb is configured to use:
            lete's proxy the logging to wandb
        Otherwise:
            return a blank function handle that does nothing
        """
        # if name == 'log':
        #     def wrapper(data, *args, **kwargs):
        #         if isinstance(data, typing.Dict):
        #             self.wandb.log(data, step=self.step)
        #         else:
        #             raise ValueError("wandb.log only supports dict")
        # else:
        return self._blank_fn if self.wandb is None else getattr(self.wandb, name)

    # def init(self):
    #     assert self.wandb is not None, "wandb is not installed"
    #     if 'entity' in self.config:
    #         entity=self.config['entity']
    #         #  TODO: wandb.init
    #     #        project=your_project_name,
    #     #        entity=your_team_name,
    #     #        notes=socket.gethostname(),
    #     #        name=your_experiment_name
    #     #        dir=run_dir,
    #     #        job_type="training",
    #     #        reinit=True)
    #     self.wandb.init( project=self.config['name'],
    #                     entity=entity,
    #                     config=self.config,

    # self.writer = None
    # self.selected_module = ""

    # if enabled:
    #     log_dir = str(log_dir)

    #     # Retrieve vizualization writer.
    #     succeeded = False
    #     for module in ["torch.utils.tensorboard", "tensorboardX"]:
    #         try:
    #             self.writer = importlib.import_module(module).SummaryWriter(log_dir)
    #             succeeded = True
    #             break
    #         except ImportError:
    #             succeeded = False
    #         self.selected_module = module

    #     if not succeeded:
    #         message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
    #             "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
    #             "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
    #         logger.warning(message)

    # self.step = 0
    # self.mode = ''

    # self.tb_writer_ftns = {
    #     'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
    #     'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
    # }
    # self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
    # self.timer = datetime.now()
