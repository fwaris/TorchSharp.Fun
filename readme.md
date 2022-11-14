# TorchSharp.Fun

A wrapper over TorchSharp that provides a **function-compostion** style of model construction. Makes the model structure more apparent. It is easier to make incremental changes to the model. Makes iterative model development faster.

## Example usage

```F#
let lisht() = 
    F [] [] (fun z ->
        use g = torch.nn.functional.tanh(z)
        z * g
    )

let model =
    torch.nn.EmbeddingBag(EMB_INDX_DIM,EMB_DIM)
    ->> torch.nn.Linear(EMB_DIM,BTL_N)
    ->> lisht()
    ->> torch.nn.Dropout(0.1)
    ->> torch.nn.Linear(BTL_N,BTL_N)
    ->> lisht()
    ->> torch.nn.Dropout(0.1)
    ->> torch.nn.Linear(BTL_N,BASE_TGTS)

```
