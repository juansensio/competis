import torch


def scale_zero_to_one(
    X:torch.Tensor, 
    dataset_min:float,
    dataset_max:float,
    ) -> torch.Tensor:
    """
    Map a tensor in range (0, 1) using prior `dataset_min`, `dataset_max`
    """
    # -> [0, 1]
    X = (X - dataset_min) / (dataset_max - dataset_min)
    return X


def undo_scale_zero_to_one(
    X:torch.Tensor, 
    dataset_min:float,
    dataset_max:float,
    ) -> torch.Tensor:
    """
    Given a tensor in range [0, 1], scale back to original dataset range [`dataset_min`, `dataset_max`]
    """
    # -> [`dataset_min`, `dataset_max`]
    S = (dataset_max - dataset_min)
    X = (S * X) + dataset_min
    return X


if __name__ == "__main__":
    X = torch.rand(30, 30)