import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Scalar, FitRes, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy
import numpy as np

class FedAvgServer(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        """Aggregate model weights using weighted average."""
        if not results:
            return None

        # Convert results
        weights_results = [
            (self.parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Aggregate weights
        aggregated_weights = self.aggregate_weights(weights_results)

        # Convert weights to parameters
        parameters_aggregated = self.weights_to_parameters(aggregated_weights)

        return parameters_aggregated, {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses and accuracies using weighted average."""
        if not results:
            return None, {}

        # Collect losses and accuracies
        losses = []
        accuracies = []
        examples = []
        for _, evaluate_res in results:
            losses.append(evaluate_res.loss)
            accuracies.append(evaluate_res.metrics["accuracy"])
            examples.append(evaluate_res.num_examples)

        # Calculate weighted averages
        avg_loss = np.average(losses, weights=examples)
        avg_accuracy = np.average(accuracies, weights=examples)

        # Print the results
        print(f"Round {rnd} - Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

        return avg_loss, {"accuracy": avg_accuracy}

    def weights_to_parameters(self, weights: List[np.ndarray]) -> Parameters:
        return fl.common.ndarrays_to_parameters(weights)

    def parameters_to_weights(self, parameters: Parameters) -> List[np.ndarray]:
        return fl.common.parameters_to_ndarrays(parameters)

    def aggregate_weights(self, results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: List[np.ndarray] = [
            np.sum(layer_updates, axis=0) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime

# Create strategy
strategy = FedAvgServer(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
)

# Start server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)