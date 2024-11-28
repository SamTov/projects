import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import neural_tangents as nt
import orbax.checkpoint
import input_pipeline
import models
import matplotlib.pyplot as plt
import glob
from train import *
import pickle
from rich.progress import track

from rich import print

def load_state(file):
    """ Load a model state. """
    with open(file, "rb") as f:
        params = pickle.load(f)

    return params

def get_apply_fn(model):

    def apply_fn(params, inputs):
        rng = jax.random.key(0)
        rng, init_rng = jax.random.split(rng)
        return model.apply(
            {"params": params}, inputs=inputs[0, :, :], rngs={'dropout': init_rng}, train=False
        )

    return apply_fn

def eval_step(params, batch):
    """Calculate evaluation metrics on a batch."""
    
    inputs, targets = batch['inputs'], batch['targets']
    weights = jnp.where(targets > 0, 1.0, 0.0)
    logits = model.apply({'params': params}, inputs=inputs, train=False)
    
    return compute_metrics(logits, targets, weights)

def compute_cvs(ntk):

    eigs, _ = np.linalg.eigh(ntk)

    eigs = np.clip(eigs, 1e-11, None)

    eigs /= eigs.sum()

    return -np.sum(eigs * np.log(eigs)), np.trace(ntk)



if __name__ == "__main__":
  
    # Get the config
    vocabs = input_pipeline.create_vocabs("ud-treebanks-v2.0/UD_Ancient_Greek/grc-ud-train.conllu")
    config = models.TransformerConfig(
    vocab_size=len(vocabs['forms']),
    output_vocab_size=len(vocabs['xpos']),
    max_len=256,
    )

    vocabs = input_pipeline.create_vocabs("ud-treebanks-v2.0/UD_Ancient_Greek/grc-ud-train.conllu")
    attributes_input = [input_pipeline.CoNLLAttributes.FORM]
    attributes_target = [input_pipeline.CoNLLAttributes.XPOS]
    train_ds = input_pipeline.sentence_dataset_dict(
        "ud-treebanks-v2.0/UD_Ancient_Greek/grc-ud-train.conllu",
        vocabs,
        attributes_input,
        attributes_target,
        batch_size=1000,
        bucket_size=config.max_len,
    )
    train_iter = iter(train_ds)
    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
    # Set rng
    rng = random.key(0)
    init_rng, key = random.split(rng)
    entropies = []
    traces = []
    loss = []

    files = np.sort(glob.glob("/work/stovey/novely-model-study/nlp_seq/model_*"))
    nums = [int(item.split("/")[-1].split("_")[-1]) for item in files]
    indices = np.argsort(nums)
    print(len(nums))
    # n_sub_samples = 50
    for item in track(files[indices][::100]):
        sub_entropies = []
        sub_traces = []
        sub_losses = []
        model = models.Transformer(config)
        params = load_state(item)
        params = jax.tree_util.tree_map(lambda x: jnp.squeeze(x), params)

        i = 0
        for train_batch in train_iter:
            
            batch = common_utils.shard(
                jax.tree_util.tree_map(lambda x: x._numpy(), train_batch)
            )
            train_keys = ['inputs', 'targets']
            (inputs, targets) = (batch.get(k, None) for k in train_keys)
            weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)
            test_ds = inputs
            logits = model.apply(
                {'params': params},
                inputs=inputs[0, :, :],
                train=False,
                rngs={'dropout': init_rng},
            )
            eval_batch = jax.tree_util.tree_map(lambda x: x._numpy(), train_batch)

            sub_losses.append(
                eval_step(params, eval_batch)
            )
            i += 1
            if i == 2:
                break

            # if i > n_sub_samples:
            #     break
            # i += 1
            
        #     ntk_fn = nt.empirical_ntk_fn(get_apply_fn(model))
        #     ntk = ntk_fn(test_ds, test_ds, params)

        #     e, t = compute_cvs(ntk)
        #     sub_entropies.append(e)
        #     sub_traces.append(t)

        # entropies.append(
        #     [np.mean(sub_entropies), np.std(sub_entropies)]
        # )
        loss.append(sub_losses)
        # traces.append(
        #     [np.mean(sub_traces), np.std(sub_traces)]
        # )

    # np.save("entropy.npy", entropies)
    # np.save("traces.npy", traces)
    np.save("losses.npy", loss)           
  