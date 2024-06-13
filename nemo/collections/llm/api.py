from pathlib import Path
from typing import Callable, Optional, Union

import pytorch_lightning as pl

from nemo.collections.llm.utils import task
from nemo.lightning import AutoResume, Experiment, MegatronStrategy, Trainer, io, teardown
from nemo.lightning.resume import Resume


@task(namespace="llm")
def train(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    exp: Experiment = Experiment(),
    resume: Optional[Union[AutoResume, Resume]] = AutoResume(),
    tokenizer: Optional[str] = None,
    # TODO: Fix export
    # export: Optional[str] = None,
) -> Path:
    """
    Trains a model using the specified data and trainer, with optional tokenizer, source, and export.

    Args:
        model (pl.LightningModule): The model to be trained.
        data (pl.LightningDataModule): The data module containing training data.
        trainer (Trainer): The trainer instance configured with a MegatronStrategy.
        exp (Experiment): An experiment instance.
        resume (Optional[Union[AutoResume, Resume]]): Resume training from a checkpoint.
        tokenizer (Optional[str]): Tokenizer setting to be applied. Can be 'data' or 'model'.
        export (Optional[str]): Filename to save the exported checkpoint after training.

    Returns
    -------
        Path: The directory path where training artifacts are saved.

    Raises
    ------
        ValueError: If the trainer's strategy is not MegatronStrategy.

    Examples
    --------
        >>> model = MyModel()
        >>> data = MyDataModule()
        >>> trainer = Trainer(strategy=MegatronStrategy())
        >>> train(model, data, trainer, tokenizer='data', source='path/to/ckpt.ckpt', export='final.ckpt')
        PosixPath('/path/to/log_dir')
    """
    if not isinstance(trainer.strategy, MegatronStrategy):
        raise ValueError("Only MegatronStrategy is supported")

    if tokenizer:  # TODO: Improve this
        _use_tokenizer(model, data, tokenizer)

    if resume is not None:
        resume.setup(model, trainer)
    app_state = exp.setup(trainer, resume_if_exists=getattr(resume, "resume_if_exists", False))

    if hasattr(train, "__io__"):
        _save_config_img(app_state.exp_dir, train.__io__)

    trainer.fit(model, data)

    # print(f"Saving checkpoint to: {export_dir}")
    # trainer.save_checkpoint(export_dir)

    # if export and trainer.strategy.is_global_zero:
    #     teardown(trainer, model=model)
    #     print(f"Exporting checkpoint to: {export_dir / export}")
    #     export_ckpt(export_dir, export)

    exp.teardown()

    return app_state.exp_dir


@task(namespace="llm")
def pretrain(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    source: Optional[str] = None,
    # export: Optional[str] = None
) -> Path:
    return train(model=model, data=data, trainer=trainer, tokenizer="data", source=source)


@task(namespace="llm")
def validate(
    model: pl.LightningModule,
    data: pl.LightningDataModule,
    trainer: Trainer,
    tokenizer: Optional[str] = None,
    source: Optional[str] = None,
    export: Optional[str] = None,
) -> Path:
    if not isinstance(trainer.strategy, MegatronStrategy):
        raise ValueError("Only MegatronStrategy is supported")

    validate_kwargs = {}
    run_dir = Path(trainer.logger.log_dir)
    export_dir = run_dir / "export"

    if tokenizer:  # TODO: Improve this
        _use_tokenizer(model, data, tokenizer)
    if source:
        _add_ckpt_path(source, model, validate_kwargs)

    trainer.validate(model, data, **validate_kwargs)
    trainer.save_checkpoint(export_dir)
    if export:
        teardown(trainer)
        del trainer, model, data
        export_ckpt(export_dir, export)

    return run_dir


@task(name="import", namespace="llm")
def import_ckpt(
    model: pl.LightningModule,
    source: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    return io.import_ckpt(model=model, source=source, output_path=output_path, overwrite=overwrite)


def load_connector_from_trainer_ckpt(path: Path, target: str) -> io.ModelConnector:
    return io.load_ckpt(path).model.exporter(target, path)


@task(name="export", namespace="llm")
def export_ckpt(
    path: Path,
    target: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    load_connector: Callable[[Path, str], io.ModelConnector] = load_connector_from_trainer_ckpt,
) -> Path:
    return io.export_ckpt(path, target, output_path, overwrite, load_connector)


def _use_tokenizer(model: pl.LightningModule, data: pl.LightningDataModule, tokenizer: str) -> None:
    if tokenizer == "data":
        model.tokenizer = data.tokenizer
    elif tokenizer == "model":
        data.tokenizer = model.tokenizer


def _add_ckpt_path(source, model, kwargs) -> None:
    if io.is_distributed_ckpt(source):
        kwargs["ckpt_path"] = source
    else:
        kwargs["ckpt_path"] = model.import_ckpt(source)


def _save_config_img(*args, **kwargs):
    try:
        from nemo_sdk.utils import save_config_img

        save_config_img(*args, **kwargs)
    except ImportError:
        pass
