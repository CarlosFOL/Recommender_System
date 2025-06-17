import implicit.als as als
import numpy as np

class ALS:
    """
    ALS model to build a recommender system.

    Attributes:
    -----------
    factors: int
        Number of latent factors
    regularization: float
        Regularization parameter
    iterations: int
        Number of iterations

    Methods
    -------
    train(interaction_matrix: np.ndarray) -> als.AlternatingLeastSquares:
        Train ALS model by using the interaction matrix.

    """

    def __init__(self, factors: int = 128, regularization: float = 0.01,
                 iterations: int = 15):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations

    def train(self, interaction_matrix: np.ndarray) -> als.AlternatingLeastSquares:
        """
        Train ALS model by using the interaction matrix

        Args:
            interaction_matrix: np.ndarray
                Interaction matrix

        Returns:
            als.AlternatingLeastSquares
                Trained ALS model
        """
        # Initialize model
        model = als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            calculate_training_loss=True
        )

        model.fit(interaction_matrix, show_progress=True)
        return model
