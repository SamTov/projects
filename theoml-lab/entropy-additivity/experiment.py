#!/usr/bin/env python
# coding: utf-8


from experiment_modules import *

n_particles = 50

name = f"/data/stovey/surrogate-variables/experiment_data"
batch_size = 100
epochs = 500

#main(ds_size=n_particles, batch_size=batch_size, epochs=epochs, name=name)


@dataclass
class Measurement:
    width: int
    depth: int
    loss: np.ndarray
    trace: np.ndarray
    entropy: np.ndarray
    representations: np.ndarray
    loss_derivatives: np.ndarray
    class_entropy: dict


results = []
    
experiments = np.load(f"{name}.npy", allow_pickle=True)

for experiment in experiments:
    generator = Generator(n_samples=n_particles)

    trace = []
    entropy = []
    class_entropy = []
    loss = []
    representations = []
    loss_derivatives = []

    model = build_network(experiment.width, experiment.depth)()
    ntk_fn = get_ntk_function(model.apply, None)

    for i in track(range(0, len(experiment.parameters))):
        predictions = jax.nn.softmax(model.apply(
            {"params": experiment.parameters[i]}, generator.train_ds["inputs"]
        ))
        # Loss computation
        loss.append(
            ((predictions - generator.train_ds["targets"]) ** 2).sum(axis=-1).mean()
        )
        representations.append(
            predictions
        )
        loss_derivatives.append(
            compute_loss_derivative(
                predictions, generator.train_ds["targets"]
                ))
        # Compute the NTK
        ntk_matrix = ntk_fn(
                    generator.train_ds["inputs"],
                    generator.train_ds["inputs"],
                    {"params": experiment.parameters[i]}
                )

        # Trace computation
        trace.append(
            compute_trace(ntk_matrix)
        )

        # Entropy computation
        entropy.append(
            compute_entropy(ntk_matrix)
        )

        # Class entropy computation
        class_entropy.append(
            compute_class_entropy(generator.train_ds, ntk_matrix)
        )
                # Class entropy computation
        class_entropy.append(
            compute_class_entropy(generator.train_ds, ntk_matrix)
        )
    results.append(
        Measurement(
            width=experiment.width,
            depth=experiment.depth,
            loss=np.array(loss),
            trace=np.array(trace),
            entropy=np.array(entropy),
            loss_derivatives=np.array(loss_derivatives),
            representations=np.array(representations),
            class_entropy=class_entropy
        )
    )
    
np.save("experiment-analysis.npy", results)
