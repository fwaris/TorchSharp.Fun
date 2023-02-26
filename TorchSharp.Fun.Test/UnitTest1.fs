module TorchSharp.Fun.Test
open NUnit.Framework
open TorchSharp
open TorchSharp.Fun
open System

[<SetUp>]
let Setup () =
    ()

[<Test>]
let CreateModel () =
    let EMB_INDX_DIM =  5L
    let EMB_DIM      = 256L
    let BTL_N        = 50L
    let BASE_TGTS    = 20L

    let create() = 
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
        let loss = torch.nn.HuberLoss()

        model,loss
    create() |> ignore
   
[<Test>]
let CreateModelWithNames() =
    let EMB_INDX_DIM =  5L
    let EMB_DIM      = 256L
    let BTL_N        = 50L
    let BASE_TGTS    = 20L

    let createNamed() = 
        let lisht() = 
            F [] [] (fun z ->
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

    createNamed() |> ignore

[<Test>]
let InvokeModel() =
    let m = torch.nn.Linear(10,10) ->> torch.nn.ReLU()
    use t1 = torch.rand([|1L;10L|], dtype = Nullable torch.float32)
    use t2 = m.forward(t1) 
    ()

[<Test>]
let SaveLoadModel() =
    let create() = torch.nn.Linear(10,10) ->> torch.nn.ReLU()
    let m1 = create()
    use t1 = torch.rand([|1L;10L|], dtype = Nullable torch.float32)
    use t2 = m1.forward(t1) 
    let tmp = System.IO.Path.GetTempFileName()
    m1.Module.save(tmp) |> ignore
    let m2 = create()
    m2.Module.load(tmp) |> ignore   
    let m1_d = m1.Module.state_dict()
    let m2_d = m2.Module.state_dict()
    let allEq = 
        Seq.zip m1_d m2_d 
        |> Seq.forall (fun (a,b) -> a.Key = b.Key && a.Value.shape = b.Value.shape)
    assert(allEq)

[<Test>]
let ``function model`` () =
    let l1 = torch.nn.Linear(10,10)
    let l2 = torch.nn.Linear(10,10)

    let model  = F [] [l1; l2] (fun t -> t --> l1 --> l2)

    use t = torch.rand([|1L;10L|], dtype=Nullable torch.float)
    use oT = model.forward(t)
    ()


[<Test>]
let ``forward with extra parameters``() = 

    //additional arguments - a wrapper around dictionary that offers some typing support
    let args = 
        (Args()
            ?extra1 <- torch.rand([|2L;10L|],dtype=Nullable torch.float))        //required argument
            ?_extraOpt1 <- torch.rand([|2L;10L|],dtype=Nullable torch.float)     //optional arg; (by convention name starts with underscore '_')

    //component modules we need 
    let l1 = torch.nn.Linear(10,10)
    let l2 = torch.nn.Linear(10,10)
    let l3 = torch.nn.Linear(10,10)
    let r1 = torch.nn.ReLU()

    let model = 
        Fx [] [l1; l2; l3; r1] (fun (tBase,args) ->
            let extra1:torch.Tensor = args?extra1                       //access additional required argument (note: type is specified at access)
            let _extraOpt1:torch.Tensor option = args?* "_extraOpt1"    //access optional argument 

            use oT = l1.forward(tBase)                                  //use the default tensor that is always passed to forward
            use oExtra1 = l2.forward(extra1)                            //use extra required argument
            let oTPlusExtra1 = oT + oExtra1
            let oFinal =                                                //final result with optional argument used, if supplied
                match _extraOpt1 with
                | Some exOp -> l3.forward(exOp) + oTPlusExtra1
                | None -> oTPlusExtra1

            let args' = args?exNorm <- oTPlusExtra1.norm()             //can add additional arugments for downstream models to use

            oFinal,args')                                              //Fx 'forward' function should return result and args

    let tBase = torch.rand([|1L;10L|],dtype=Nullable torch.float)
    let tF,arg2 = model.forward(tBase,args)                            //invoke the model with extra arguments
    ()