import os
import json
import torch
from torchmin import minimize


def exponential_velocity_decay(
    time, initial_velocity, initial_position, velocity_decay_factor
):
    position = [initial_position.reshape((1, 2))]
    velocity = [initial_velocity.reshape((1, 2))]
    for i in range(len(time) - 1):
        position.append(position[i] + velocity[i] * (time[i + 1] - time[i]))
        velocity.append(velocity[i] * velocity_decay_factor)
    return torch.concatenate(position)


def estimate_initial_velocity(time, positions, n_points=3):
    velocity = torch.zeros(n_points, 2)
    for i in range(n_points):
        velocity[i, :] = (positions[i + 1] - positions[i]) / (time[i + 1] - time[i])
    return torch.mean(velocity, dim=0)


if __name__ == "__main__":
    data_path = "data/"
    file_list = os.listdir(data_path)
    for file in file_list:
        file_path = data_path + file
        print(file)
        with open(file_path, "r") as f:
            json_data = json.load(f)
            positions = torch.tensor(
                [json_data[i][0]["position"] for i in range(len(json_data))]
            )
            time = torch.tensor(
                [json_data[i][0]["time"] for i in range(len(json_data))]
            )
            time = time - time[0]

            def objective_function(velocity_decay_factor):
                # Apply model to predict data
                # Calculate initial velocity
                initial_velocity = estimate_initial_velocity(time, positions, 4)
                print(initial_velocity)
                initial_position = positions[0]
                predicted_positions = exponential_velocity_decay(
                    time, initial_velocity, initial_position, velocity_decay_factor
                )
                residuals = predicted_positions - positions
                errors = torch.sqrt(torch.sum(residuals**2, dim=1))
                loss = errors.mean()
                print(loss)
                return loss

            initial_velocity_decay_factor = 1.0
            result = minimize(
                objective_function,
                initial_velocity_decay_factor,
                method="bfgs",
                tol=1e-9,
            )
            print(result)
        break
