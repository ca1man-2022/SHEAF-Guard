"""Toy tensors only: no learned parameters, embeddings or real queries."""
import json
import numpy as np
from sheaf_guard_preview import extension, compatibility_readout


def toy_inputs():
    return (np.array([0.25, -0.5]),
            np.array([[[1., .2], [0., .8]], [[.7, 0.], [.1, 1.2]]]),
            np.array([[[.1, -.2], [.9, .4]], [[-.1, -.3], [.6, .8]]]), .7)


def main():
    result = extension(*toy_inputs(), weights=[.5, .5])
    margin, minimum = compatibility_readout(result["energies"])
    print(json.dumps(dict(input_kind="synthetic", E_ID=float(result["energies"][0]),
                          E_OOD=float(result["energies"][1]),
                          margin=float(margin), minimum=float(minimum)), indent=2))


if __name__ == "__main__":
    main()
