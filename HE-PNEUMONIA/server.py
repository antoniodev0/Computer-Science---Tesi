import flwr as fl
from flwr.server.strategy import FedAvg
import tenseal as ts
import numpy as np
import pickle

# Carica il contesto TenSEAL
with open("public_context.pkl", "rb") as f:
    public_context = pickle.load(f)

context = ts.context_from(public_context)

class HomomorphicFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        encrypted_params = []
        weights = []
        for client, fit_res in results:
            client_params = []
            for param in fl.common.parameters_to_ndarrays(fit_res.parameters):
                ckks_vector = ts.lazy_ckks_vector_from(param.tobytes())
                ckks_vector.link_context(context)
                
                # Accedi a partition_id dalle metrics
                partition_id = fit_res.metrics["partition_id"]
                print(f"Encrypted parameter received on server (partition_id {partition_id}): {ckks_vector}")

                client_params.append(ckks_vector)
            encrypted_params.append(client_params)
            weights.append(fit_res.num_examples)

        total_examples = sum(weights)
        normalized_weights = [w / total_examples for w in weights]

        aggregated_params = [param * normalized_weights[0] for param in encrypted_params[0]]
        for client_params, weight in zip(encrypted_params[1:], normalized_weights[1:]):
            for i in range(len(aggregated_params)):
                aggregated_params[i] += client_params[i] * weight

        aggregated_serialized = [param.serialize() for param in aggregated_params]
        return fl.common.ndarrays_to_parameters(aggregated_serialized), {}

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        # Aggrega la perdita (val_loss) e l'accuratezza in chiaro dei client
        total_loss = sum(r.metrics["val_loss"] * r.num_examples for _, r in results)
        total_accuracy = sum(r.metrics["accuracy"] * r.num_examples for _, r in results)
        total_samples = sum(r.num_examples for _, r in results)

        average_loss = total_loss / total_samples
        average_accuracy = total_accuracy / total_samples

        print(f"Round {server_round} - Average loss: {average_loss:.4f}, Average accuracy: {average_accuracy:.4f}")

        return average_loss, {"accuracy": average_accuracy}

def main():
    strategy = HomomorphicFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()