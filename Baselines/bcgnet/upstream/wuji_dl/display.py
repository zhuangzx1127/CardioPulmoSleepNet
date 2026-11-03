import pandas as pd
import rich.table
import typing as t
import wandb


class DefaultStyler:
    def __init__(self, float_format: t.Optional[str] = None) -> None:
        self.float_format = float_format

    def __call__(self, x: t.Any) -> str:
        if isinstance(x, float) and self.float_format is not None:
            return self.float_format.format(x)
        else:
            return str(x)


def df_to_rich(
    df: pd.DataFrame,
    corner: str = "",
    styler: t.Callable[[t.Any], str] = DefaultStyler(float_format="{:.3}"),
) -> rich.table.Table:
    table = rich.table.Table()
    table.add_column(corner, style="cyan", justify="center")
    for col in df.columns:
        table.add_column(col, style="magenta", justify="center")
    for row in df.itertuples():
        table.add_row(*(styler(x) for x in row))
    return table


def df_to_wandb(df: pd.DataFrame, corner: str = "") -> wandb.Table:
    table = wandb.Table(columns=[corner, *df.columns])
    for row in df.itertuples():
        table.add_data(*row)
    return table
