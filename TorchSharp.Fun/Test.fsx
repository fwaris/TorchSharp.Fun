#r "nuget: FsPickler"
#r "nuget: TorchSharp-cpu" 
#load "TorchSharp.Fun.fs"

open TorchSharp.Fun
open TorchSharp
open type TorchSharp.torch.nn

module EmbNet =
    let EMB_INDX_DIM =  5L
    let EMB_DIM      = 256L
    let BTL_N        = 50L
    let BASE_TGTS    = 20L

    let create() = 
        let lisht() = 
            F [] [] [] (fun z ->
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
        let loss = torch.nn.HuberLoss()

        model,loss

    let createNamed() = 
        let lisht() = 
            F [] [] [] (fun z ->
                use g = torch.nn.functional.tanh(z)
                z * g
            )

        let model =
            torch.nn.EmbeddingBag(EMB_INDX_DIM,EMB_DIM)
            =>> ("emb",torch.nn.Linear(EMB_DIM,BTL_N))
            =>> ("ls",lisht())
            =>> ("drop", torch.nn.Dropout(0.1))
            =>> ("proj1",torch.nn.Linear(BTL_N,BTL_N))
            =>> ("ls2",lisht())
            =>> ("drop2",torch.nn.Dropout(0.1))
            =>> ("proj20",torch.nn.Linear(BTL_N,BASE_TGTS))

        let loss = torch.nn.HuberLoss()

        model,loss

module EmbNetTx =
    let EMB_INDX_DIM = 100L  * 5L
    let EMB_DIM      = 128L
    let BTL_N        = 10L        
    let BASE_TGTS    = 20L

    let create() = 
        let lisht() = 
            F [] [] [] (fun z ->
                use g =  torch.nn.functional.tanh(z)
                z * g
            )
            
        let encL      = TransformerEncoderLayer(EMB_DIM,8L)
        let enc       = TransformerEncoder(encL,2L)
        let embedding = torch.nn.Embedding(EMB_INDX_DIM,EMB_DIM)
        let proj      = torch.nn.Linear(EMB_DIM,BASE_TGTS)
        let ls        = torch.nn.SiLU() //lisht

        let ``:`` = torch.TensorIndex.Colon
        let first = torch.TensorIndex.Single(0L)

        let model = Fx ["model"; "tx"; "proj"; "emb"; "ls"] [M enc; M proj; M embedding; M ls] [] (fun (tokenIds,args) ->
            use emb         = embedding.forward(tokenIds)
            use embB2nd     = emb.permute(1L,0L,2L)
            use encoded     = enc.forward(embB2nd) 
            use encodedB1st = encoded.permute(1L,0L,2L)
            use pooled      = encodedB1st.index(``:``,first)
            use pooledact   = ls.forward(pooled) 
            proj.forward(pooledact),args)

        let loss = torch.nn.MSELoss()

        model,loss

let a = torch.nn.EmbeddingBag(5L,5L)
let b = torch.nn.Dropout()
let c = a ->> b
c.Module.GetName()
c.Module.named_children() |> Seq.iter (fun struct(n,x) -> printfn "%A" (n,x))

let m1,l1 = EmbNet.createNamed()
m1.Module.named_children() |> Seq.iter (fun struct(n,x) -> printfn "%A" (n,x))

let m2,l2 = EmbNet.create()
m2.forward(torch.tensor([|1L;2L|],dimensions=[|1L;1L|]), Args())
m2.Module.named_children() |> Seq.iter (fun struct(n,x) -> printfn "%A" (n,x))
