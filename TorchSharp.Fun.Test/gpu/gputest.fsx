#r "nuget: FsPickler"
#r "nuget: TorchSharp-cuda-windows" 
#load "../../TorchSharp.Fun/TorchSharp.Fun.fs"

open TorchSharp.Fun
open TorchSharp

let device = if torch.cuda_is_available() then torch.CUDA else failwith "this test needs cuda"

//component modules we need 

(*
With recent PyTorch / TorchSharp changes (© 0.99) buffers / parameters in func models are not moved 
'to' target device as they are not fields of the Module instance
to address, we need ref parameters and need to use the to'(device) extension
method to also move the buffers.
However this is a hack that does not work if the FuncModel is nested a few layers down
*)

let m1 =
    let l1 = torch.nn.Linear(10,10)
    let d = torch.nn.Dropout()
    let buf = new Modules.Parameter( torch.ones([|10L;10L|]),requires_grad=false)
    buf.name <- "buf"
    let bufRef = ref buf
    F [] [l1; d; bufRef] (fun t -> (t --> l1 --> d) + bufRef.Value)

let m2 = 
    torch.nn.Linear(10,10)
    ->> torch.nn.Linear(10,10)
    ->> torch.nn.ReLU()

let m = m1 ->> m2

m.to'(device)

let t1input = torch.rand([|10L|]).``to``(device)
let t' = m.forward t1input
