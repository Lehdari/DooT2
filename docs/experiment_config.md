Experiment Config JSON specification
====================================

This file describes the experiment configuration JSON file format and it's intended usage.

Mandatory Entries
-----------------
Entries in the JSON without which the configuration is considered to be invalid.

- `experiment_root`: Path to the experiment directory w.r.t. the experiments directory (experiments directory
  defaults to `experiments/` and can be overridden by CLI arguments or the GUI).
- `model_type`: Trained model type name
- `model_config`: JSON subobject containing all the necessary information for constructing the model subject to
  training, including torch model paths, hyperparameters etc. It is the only field inside `experimentConfig` that
  is allowed to be modified by `Model` (and derivative) instances.
- `software_version`: Git version hash of the software that was used to run the experiment. For example:
  `6da8e26-dirty`
