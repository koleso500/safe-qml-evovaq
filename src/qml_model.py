import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


def apply_quantum_noise(noise_type: str, noise_prob: float, n_qubits: int):
    if noise_prob <= 0:
        return

    if noise_type == 'depolarizing':
        for wire in range(n_qubits):
            qml.DepolarizingChannel(noise_prob, wires=wire)

    elif noise_type == 'bit_flip':
        for wire in range(n_qubits):
            qml.BitFlip(noise_prob, wires=wire)

    elif noise_type == 'phase_flip':
        for wire in range(n_qubits):
            qml.PhaseFlip(noise_prob, wires=wire)

    elif noise_type == 'amplitude_damping':
        for wire in range(n_qubits):
            qml.AmplitudeDamping(noise_prob, wires=wire)

    else:
        raise ValueError(
            f'Unknown noise_type={noise_type!r}. '
            'Use one of: depolarizing, bit_flip, phase_flip, amplitude_damping.'
        )


def make_vqc_circuit(
    n_qubits: int,
    noise_enabled: bool = False,
    noise_type: str = 'depolarizing',
    noise_prob: float = 0.0
):
    device_name = 'default.mixed' if noise_enabled else 'default.qubit'
    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface='torch')
    def vqc_circuit(inputs, weights):
        qml.AmplitudeEmbedding(
            inputs,
            wires=range(n_qubits),
            normalize=True
        )

        qml.StronglyEntanglingLayers(
            weights,
            wires=range(n_qubits)
        )

        if noise_enabled:
            apply_quantum_noise(
                noise_type=noise_type,
                noise_prob=noise_prob,
                n_qubits=n_qubits
            )

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return vqc_circuit, dev


class QuantumClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        n_qubits: int,
        n_layers: int,
        noise_enabled: bool = False,
        noise_type: str = 'depolarizing',
        noise_prob: float = 0.0
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise_enabled = noise_enabled
        self.noise_type = noise_type
        self.noise_prob = noise_prob

        self.vqc_circuit, self.q_device = make_vqc_circuit(
            n_qubits=n_qubits,
            noise_enabled=noise_enabled,
            noise_type=noise_type,
            noise_prob=noise_prob
        )

        weight_shapes = {
            'weights': (n_layers, n_qubits, 3)
        }

        self.pre_pqc = nn.Linear(input_dim, 2 ** n_qubits)
        self.q_layer = qml.qnn.TorchLayer(
            self.vqc_circuit,
            weight_shapes
        )
        self.fc = nn.Linear(n_qubits, num_classes)

    def forward(self, inputs):
        inputs = self.pre_pqc(inputs)
        inputs = torch.nn.functional.gelu(inputs)
        inputs = self.q_layer(inputs)
        return self.fc(inputs)

    def set_quantum_weights_flat(self, flat_params):
        weights = np.asarray(flat_params, dtype=np.float32).reshape(
            self.n_layers,
            self.n_qubits,
            3
        )

        with torch.no_grad():
            self.q_layer.weights.copy_(
                torch.tensor(
                    weights,
                    dtype=self.q_layer.weights.dtype,
                    device=self.q_layer.weights.device
                )
            )


def get_quantum_weights_flat(model: QuantumClassifier):
    return model.q_layer.weights.detach().cpu().numpy().reshape(-1)


def predict_proba_qml(model, x_data, device):
    model.eval()

    with torch.no_grad():
        logits = model(
            torch.tensor(
                x_data,
                dtype=torch.float32,
                device=device
            )
        )
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs