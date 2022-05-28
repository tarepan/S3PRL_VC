"""Configuration loader generation"""


from typing import Callable, TypeVar

from omegaconf import OmegaConf, SCMode


T = TypeVar('T') # dataclass
def generate_conf_loader(default_str: str, conf_class: T) -> Callable[[], T]:
    """Generate 'Load configuration type-safely' function.
    Priority: CLI args > CLI-specified config yaml > Default
    """

    def load_configuration() -> T:
        """Load configurations."""
        default = OmegaConf.create(default_str)
        cli = OmegaConf.from_cli()

        extends_path = cli.get("path_extend_conf", None)
        if extends_path:
            extends = OmegaConf.load(extends_path)
            conf_final = OmegaConf.merge(default, extends, cli)
        else:
            conf_final = OmegaConf.merge(default, cli)
        OmegaConf.resolve(conf_final)
        conf_structured = OmegaConf.merge(
            OmegaConf.structured(conf_class),
            conf_final
        )

        # Design Note -- OmegaConf instance v.s. DataClass instance --
        #   OmegaConf instance has runtime overhead in exchange for type safety.
        #   Configuration is constructed/finalized in early stage,
        #   so config is eternally valid after validation in last step of early stage.
        #   As a result, we can safely convert OmegaConf to DataClass after final validation.
        #   This prevent (unnecessary) runtime overhead in later stage.
        #
        #   One demerit: No "freeze" mechanism in instantiated dataclass.
        #   If OmegaConf, we have `OmegaConf.set_readonly(conf_final, True)`

        # `.to_container()` with `SCMode.INSTANTIATE` resolve interpolations and check MISSING.
        # It is equal to whole validation.
        return OmegaConf.to_container(conf_structured, structured_config_mode=SCMode.INSTANTIATE) # type: ignore ; It is validated by omegaconf

    return load_configuration
