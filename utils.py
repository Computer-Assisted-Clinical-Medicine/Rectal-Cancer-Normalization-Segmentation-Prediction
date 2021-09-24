"""
Miscellaneous functions used mainly for plotting
"""
import logging
import os
import stat
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from SegmentationNetworkBasis.architecture import DenseTiramisu, UNet, DeepLabv3plus

# if on cluster, use other backend
# pylint: disable=wrong-import-position, ungrouped-imports
if "CLUSTER" in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-order


def configure_logging(tf_logger: logging.Logger) -> logging.Logger:
    """Configure the logger, the handlers of the tf_logger are removed and both
    loggers are set to

    Parameters
    ----------
    tf_logger : logging.Logger
        The tensorflow logger, must be assigned before importing tensorflow

    Returns
    -------
    logging.Logger
        The base logger
    """
    # configure loggers
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    tf_logger.setLevel(logging.DEBUG)
    # there is too much output otherwise
    for handler in tf_logger.handlers:
        tf_logger.removeHandler(handler)
    return logger


def plot_hparam_comparison(
    experiment_dir: os.PathLike,
    metrics=None,
    external=False,
    postprocessed=False,
    version="final",
    plot_errors=True,
):
    """
    Do separate plots for every changed hyperparameter.
    """
    experiment_dir = Path(experiment_dir)

    if metrics is None:
        metrics = ["Dice"]

    hparam_file = experiment_dir / "hyperparameters.csv"
    hparam_changed_file = experiment_dir / "hyperparameters_changed.csv"

    result_name = f"hy_comp_{version}"
    if external:
        result_name += "_external_testset"
    if postprocessed:
        result_name += "_postprocessed"
    # add pdf
    result_name += ".pdf"

    hparams: pd.DataFrame = pd.read_csv(hparam_file, sep=";")
    hparams_changed: pd.DataFrame = pd.read_csv(hparam_changed_file, sep=";")
    changed_params = hparams_changed.columns[1:]

    res_path = generate_res_path(version, external, postprocessed)

    # type is incorrectly detected
    # pylint: disable=no-member,unsubscriptable-object

    # collect mean and error of mean from all results
    results_means_collected = []
    results_m_err_collected = []
    found_any = False
    for exp_loc in hparams["path"]:
        results_file = experiment_dir / exp_loc / res_path
        if results_file.exists():
            results = pd.read_csv(results_file, sep=";")
            # save results
            results_means_collected.append(results[metrics].mean())
            results_m_err_collected.append(
                results[metrics].std() / np.sqrt(results.shape[0])
            )
            found_any = True
        else:
            name = results_file.parent.parent.name
            print(
                f"Could not find the evaluation file for {name}"
                + " (probably not finished with training yet)."
            )
            results_means_collected.append(pd.Series({m: pd.NA for m in metrics}))
            results_m_err_collected.append(pd.Series({m: pd.NA for m in metrics}))

    if not found_any:
        print("No files were found")
        return

    # convert to dataframes
    results_means = pd.DataFrame(results_means_collected)
    results_m_errs = pd.DataFrame(results_m_err_collected)

    # plot all metrics with all parameters
    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(changed_params),
        sharey=True,
        figsize=(4 * len(changed_params), 6 * len(metrics)),
    )
    # fix the dimensions
    axes = np.array(axes).reshape((len(metrics), len(changed_params)))

    # fix the names
    for col in hparams_changed:
        if not pd.api.types.is_numeric_dtype(hparams_changed[col]):
            hparams_changed[col] = hparams_changed[col].fillna("None").astype(str)

    for met, ax_row in zip(metrics, axes):
        for col, ax in zip(changed_params, ax_row):
            # group by the other values
            unused_columns = [cn for cn in changed_params if col != cn]
            # if there are no unused columns, use the changed one
            if len(unused_columns) == 0:
                unused_columns = list(changed_params)
            for group, data in hparams_changed.groupby(unused_columns, dropna=False):
                # plot them with the same line
                # get the data
                m_data = results_means.loc[data.index, met]
                # sort by values
                m_data.sort_values()
                # get standard deviations
                if plot_errors:
                    m_data_err = results_m_errs.loc[m_data.index, met]
                else:
                    m_data_err = None
                # only plot if not nan
                if not m_data.isna().all():
                    _, caps, bars = ax.errorbar(
                        x=data.loc[m_data.notna(), col],
                        y=m_data[m_data.notna()],
                        yerr=m_data_err[m_data.notna()],
                        label=str(group),
                        marker="x",
                        capsize=4,
                    )
                    # loop through bars and caps and set the alpha value
                    for bar in bars:
                        bar.set_alpha(0.4)
                    for cap in caps:
                        cap.set_alpha(0.4)
            # if the label is text, turn it
            if not pd.api.types.is_numeric_dtype(hparams_changed[col]):
                plt.setp(
                    ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
                )
            # ylabel if it is the first image
            if col == changed_params[0]:
                ax.set_ylabel(met)
            # xlabel if it is the last row
            if met == metrics[-1]:
                ax.set_xlabel(col)
            # if the class is bool, replace the labels with the boolean values
            if isinstance(hparams_changed.iloc[0][col], np.bool_):
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["false", "true"])

            # set the legend with title
            ax.legend(title=str(tuple(str(c)[:5] for c in unused_columns)))

    fig.suptitle("Hypereparameter Comparison")
    plt.tight_layout()
    result_dir = experiment_dir / "analysis" / "hyperparameter_comparison"
    if not result_dir.exists():
        result_dir.mkdir(parents=True)
    plt.savefig(result_dir / result_name)
    plt.close()


def generate_res_path(version: str, external: bool, postprocessed: bool):
    """For a given path, generate the relative path to the result file"""
    if external:
        folder_name = f"results_external_testset_{version}"
    else:
        folder_name = f"results_test_{version}"
    if postprocessed:
        folder_name += "-postprocessed"
    res_path = Path(folder_name) / "evaluation-all-files.csv"
    return res_path


def compare_hyperparameters(experiments, experiment_dir, version="best"):
    """
    Compare the hyperparameters of all experiments and collect the ones that
    were changed.
    """
    # export the hyperparameters
    hyperparameter_file = experiment_dir / "hyperparameters.csv"
    hyperparameter_changed_file = experiment_dir / "hyperparameters_changed.csv"
    # collect all results
    hparams = []
    for exp in experiments:
        # and parameters
        hparams.append(
            {
                **exp.hyper_parameters["network_parameters"],
                **exp.hyper_parameters["train_parameters"],
                "normalizing_method": exp.hyper_parameters["preprocessing_parameters"][
                    "normalizing_method"
                ],
                "loss": exp.hyper_parameters["loss"],
                "architecture": exp.hyper_parameters["architecture"].__name__,
                "dimensions": exp.hyper_parameters["dimensions"],
                "path": exp.output_path_rel,
            }
        )

    # convert to dataframes
    hparams = pd.DataFrame(hparams)
    # find changed parameters
    changed_params = []
    # drop the results file when analyzing the changed hyperparameters
    for col in hparams:
        if hparams[col].astype(str).unique().size > 1:
            changed_params.append(col)
    # have at least one changed parameters (for the plots)
    if len(changed_params) == 0:
        changed_params = ["architecture"]
    hparams_changed = hparams[changed_params].copy()
    # if n_filters, use the first
    if "n_filters" in hparams_changed:
        hparams_changed.loc[:, "n_filters"] = (
            hparams_changed["n_filters"].dropna().apply(lambda x: x[0])
        )
    if "normalizing_method" in hparams_changed:
        n_name = hparams_changed["normalizing_method"].apply(lambda x: x.name)
        hparams_changed.loc[:, "normalizing_method"] = n_name
    # ignore the batch size (it correlates with the dimension)
    if "batch_size" in hparams_changed:
        hparams_changed.drop(columns="batch_size", inplace=True)
    # ignore do_bias (it is set the opposite to batch_norm)
    if "do_bias" in hparams_changed and "do_batch_normalization" in hparams_changed:
        hparams_changed.drop(columns="do_bias", inplace=True)
    # drop column specifying the files
    if "path" in hparams_changed:
        hparams_changed.drop(columns="path", inplace=True)
    # drop columns only related to architecture
    if "architecture" in hparams_changed:
        arch_groups = hparams_changed.groupby("architecture")
        if arch_groups.ngroups > 1:
            arch_params = arch_groups.nunique(dropna=False)
            for col in arch_params:
                if np.all(arch_params[col] == 1):
                    hparams_changed.drop(columns=col, inplace=True)

    hparams.to_csv(hyperparameter_file, sep=";")
    hparams_changed.to_csv(hyperparameter_changed_file, sep=";")


def generate_folder_name(parameters):
    """
    Make a name summarizing the hyperparameters.
    """
    epochs = parameters["train_parameters"]["epochs"]

    params = [
        parameters["architecture"].get_name() + str(parameters["dimensions"]) + "D",
        parameters["loss"],
    ]

    # TODO: move this logic into the network
    if parameters["architecture"] is UNet:
        # attention parameters
        if "encoder_attention" in parameters["network_parameters"]:
            if parameters["network_parameters"]["encoder_attention"] is not None:
                params.append(parameters["network_parameters"]["encoder_attention"])
        if "attention" in parameters["network_parameters"]:
            if parameters["network_parameters"]["attention"]:
                params.append("Attn")

        # residual connections if it is an attribute
        if "res_connect" in parameters["network_parameters"]:
            if parameters["network_parameters"]["res_connect"]:
                params.append("Res")
            else:
                params.append("nRes")

        # filter multiplier
        params.append("f_" + str(parameters["network_parameters"]["n_filters"][0] // 8))

        # batch norm
        if parameters["network_parameters"]["do_batch_normalization"]:
            params.append("BN")
        else:
            params.append("nBN")

        # dropout
        if parameters["network_parameters"]["drop_out"][0]:
            params.append("DO")
        else:
            params.append("nDO")
    elif parameters["architecture"] is DenseTiramisu:
        params.append("gr_" + str(parameters["network_parameters"]["growth_rate"]))

        params.append(
            "nl_" + str(len(parameters["network_parameters"]["layers_per_block"]))
        )
    elif parameters["architecture"] is DeepLabv3plus:
        params.append(str(parameters["network_parameters"]["backbone"]))

        params.append(
            "aspp_"
            + "_".join([str(n) for n in parameters["network_parameters"]["aspp_rates"]])
        )
    else:
        raise NotImplementedError(f'{parameters["architecture"]} not implemented')

    # normalization
    params.append(str(parameters["preprocessing_parameters"]["normalizing_method"].name))

    # object fraction
    params.append(
        f'obj_{int(parameters["train_parameters"]["percent_of_object_samples"]*100):03d}%'
    )

    # add epoch number
    params.append(str(epochs))

    folder_name = "-".join(params)

    return folder_name


def gather_results(
    experiment_dir: os.PathLike,
    external=False,
    postprocessed=False,
    combined=True,
    version="best",
) -> pd.DataFrame:
    """Collect all result files from all experiments. Only experiments that are
    already finished will be included in the analysis.

    Parameters
    ----------
    experiment_dir : Pathlike
        The path where the experiments are located
    external : bool, optional
        If the external testset should be evaluated, by default False
    postprocessed : bool, optional
        If the data from the posprocessed should be evaluated, by default False
    combined : bool, optional
        If there is a combined model, which should be analyzed, by default True
    version : str, optional
        Which version of the model should be used, by default best

    Returns
    -------
    pd.DataFrame
        The results with all metrics for all files
    """
    experiment_dir = Path(experiment_dir)
    hparam_file = experiment_dir / "hyperparameters.csv"

    if external:
        file_field = "results_file_external_testset"
    else:
        file_field = "results_file"

    if postprocessed:
        file_field += "_postprocessed"

    res_path = generate_res_path(version, external, postprocessed)

    hparams = pd.read_csv(hparam_file, sep=";")
    # type is incorrectly detected
    # pylint: disable=no-member

    # add combined model if present
    if combined:
        c_path = "combined_models"
        loc = hparams.shape[0]
        hparams.loc[loc] = "Combined"
        hparams.loc[loc, "path"] = c_path

    results_all_list = []
    for _, row in hparams.iterrows():
        results_file = experiment_dir / row["path"] / res_path
        if results_file.exists():
            results = pd.read_csv(results_file, sep=";")
            # set the model
            results["name"] = Path(row["path"]).name
            # save results
            results_all_list.append(results)
        else:
            name = Path(results_file).parent.parent.name
            print(
                f"Could not find the evaluation file for {name}"
                + " (probably not finished with training yet)."
            )

    if len(results_all_list) == 0:
        print("No files found")
        return None
    else:
        results_all = pd.concat(results_all_list)
    # drop first column (which is just the old index)
    results_all.drop(results_all.columns[0], axis="columns", inplace=True)
    results_all["fold"] = pd.Categorical(results_all["fold"])
    results_all["name"] = pd.Categorical(results_all["name"])
    results_all.index = pd.RangeIndex(results_all.shape[0])
    results_all.sort_values("File Number", inplace=True)
    return results_all


def export_slurm_job(
    filename,
    command,
    job_name=None,
    workingdir=None,
    venv_dir="venv",
    job_type="CPU",
    cpus=1,
    hours=0,
    minutes=30,
    log_dir=None,
    log_file=None,
    error_file=None,
    array_job=False,
    array_range="0-4",
    singleton=False,
    variables=None,
):
    """Generates a slurm file to run jobs on the cluster

    Parameters
    ----------
    filename : Path or str
        Where the slurm file should be saved
    command : str
        The command to run (can also be multiple commands separated by line breaks)
    job_name : str, optional
        The name displayed in squeue and used for log_name, by default None
    workingdir : str, optional
        The directory in Segmentation_Experiment, if None, basedir is used, by default None
    venv_dir : str, optional
        The directory of the virtual environment, by default venv
    job_type : str, optional
        type of job, CPU, GPU or GPU_no_K80, by default 'CPU'
    cpus : int, optional
        number of CPUs, by default 1
    hours : int, optional
        Time the job should run in hours, by default 0
    minutes : int, optional
        Time the job should run in minutes, by default 30
    log_dir : str, optional
        dir where the logs should be saved if None logs/job_name/, by default None
    log_file : str, optional
        name of the log file, if None job_name_job_id_log.txt, by default None
    error_file : str, optional
        name of the errors file, if None job_name_job_id_log_errors.txt, by default None
    array_job : bool, optional
        If set to true, array_range should be set, by default False
    array_range : str, optional
        array_range as str (comma separated or start-stop (ends included)), by default '0-4'
    singleton : bool, optional
        if only one job with that name and user should be running, by default False
    variables : dict, optional
        environmental variables to write {name : value} $EXPDIR can be used, by default {}
    """

    if variables is None:
        variables = {}

    # this new node dos not work
    exclude_nodes = ["h08c0301", "h08c0401", "h08c0501"]
    if job_type == "GPU_no_K80":
        exclude_nodes += [
            "h05c0101",
            "h05c0201",
            "h05c0301",
            "h05c0401",
            "h05c0501",
            "h06c0301",
            "h05c0601",
            "h05c0701",
            "h05c0801",
            "h05c0901",
            "h06c0101",
            "h06c0201",
            "h06c0401",
            "h06c0501",
            "h06c0601",
            "h06c0701",
            "h06c0801",
            "h06c0901",
        ]

    if job_type == "CPU":
        assert hours == 0
        assert minutes <= 30
    else:
        assert minutes < 60
        assert hours <= 48

    if log_dir is None:
        log_dir = Path("logs/{job_name}/")
    else:
        log_dir = Path(log_dir)

    if log_file is None:
        if array_job:
            log_file = log_dir / f"{job_name}_%a_%A_log.txt"
        else:
            log_file = log_dir / f"{job_name}_%j_log.txt"
    else:
        log_file = log_dir / log_file

    if error_file is None:
        if array_job:
            error_file = log_dir / f"{job_name}_%a_%A_errors.txt"
        else:
            error_file = log_dir / f"{job_name}_%j_errors.txt"
    else:
        error_file = log_dir / error_file

    filename = Path(filename)

    slurm_file = "#!/bin/bash\n\n"
    if job_name is not None:
        slurm_file += f"#SBATCH --job-name={job_name}\n"

    slurm_file += f"#SBATCH --cpus-per-task={cpus}\n"
    slurm_file += "#SBATCH --ntasks-per-node=1\n"
    slurm_file += f"#SBATCH --time={hours:02d}:{minutes:02d}:00\n"
    slurm_file += "#SBATCH --mem=32gb\n"

    if job_type in ("GPU", "GPU_no_K80"):
        slurm_file += "\n#SBATCH --partition=gpu-single\n"
        slurm_file += "#SBATCH --gres=gpu:1\n"

    if len(exclude_nodes) > 0:
        slurm_file += "#SBATCH --exclude=" + ",".join(exclude_nodes) + "\n"

    if array_job:
        slurm_file += f"\n#SBATCH --array={array_range}\n"

    # add logging
    slurm_file += f"\n#SBATCH --output={str(log_file)}\n"
    slurm_file += f"#SBATCH --error={str(error_file)}\n"

    if singleton:
        slurm_file += "\n#SBATCH --dependency=singleton\n"

    # define workdir, add diagnostic info
    slurm_file += """
echo "Set Workdir"
WSDIR=/gpfs/bwfor/work/ws/hd_mo173-myws
echo $WSDIR
EXPDIR=$WSDIR\n"""

    # print task ID depending on type
    if array_job:
        slurm_file += '\necho "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID\n'
    else:
        slurm_file += '\necho "My SLURM_JOB_ID: " $SLURM_JOB_ID\n'

    slurm_file += """\necho "job started on Node: $HOSTNAME"

echo "Load modules"

module load devel/python_intel/3.7
"""

    # add environmental variables
    if len(variables) > 0:
        slurm_file += "\n"
    for key, val in variables.items():
        slurm_file += f'export {key}="{val}"\n'

    if "GPU" in job_type:
        slurm_file += """module load devel/cuda/10.1
module load lib/cudnn/7.6.5-cuda-10.1

echo "Get GPU info"
nvidia-smi
"""

    slurm_file += '\necho "Go to workingdir"\n'
    if workingdir is None:
        slurm_file += "cd $EXPDIR/nnUNet\n"
    else:
        slurm_file += f"cd {Path(workingdir).resolve()}\n"

    # activate virtual environment
    slurm_file += '\necho "Activate virtual environment"\n'
    slurm_file += f"source {Path(venv_dir).resolve()}/bin/activate\n"

    # run the real command
    slurm_file += '\necho "Start calculation"\n\n'
    slurm_file += command
    slurm_file += '\n\necho "Finished"'

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
    # write to file
    with open(filename, "w+") as f:
        f.write(slurm_file)


def export_batch_file(filename, commands):
    """Exports a list of commands (one per line) as batch script

    Parameters
    ----------
    filename : str or Path
        The new file
    commands : [str]
        List of commands (as strings)
    """

    filename = Path(filename)

    batch_file = "#!/bin/bash"

    for com in commands:
        batch_file += f"\n\n{com}"

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)
    # write to file
    with open(filename, "w+") as f:
        f.write(batch_file)

    # set permission
    os.chmod(filename, stat.S_IRWXU)
