import pandas as pd
import numpy as np

class DataLoader:
    def __init__(
        self,
        df: pd.DataFrame,
        attribute_variables: list[str],
        target_variable: str,
        test_size: float = 0.2,
        random_state: int | None = 0,
    ) -> None:
        assert target_variable not in attribute_variables
        assert 0.0 < test_size < 1.0

        shuffled_df = df[attribute_variables + [target_variable]].sample(
            frac=1, random_state=random_state
        )

        self.X = shuffled_df[attribute_variables]
        self.y = shuffled_df[target_variable]


    def get_partitions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X.to_numpy(), self.y.to_numpy()

