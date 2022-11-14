#r "nuget: FsPickler"
#r "nuget: TorchSharp-cuda-windows" 
#load "../../TorchSharp.Fun/TorchSharp.Fun.fs"

open TorchSharp.Fun
open TorchSharp

let device = if torch.cuda_is_available() then torch.CUDA else torch.CPU

//component modules we need 

let m1 =
   let l1 = torch.nn.Linear(10,10)
   let d = torch.nn.Dropout()
   F [] [M l1; M d] (fun t -> t --> l1 --> d)

let m2 = 
    torch.nn.Linear(10,10)
    ->> torch.nn.Linear(10,10)
    ->> torch.nn.ReLU()

let m = m1 ->> m2

m.Module.``to``(device)

let t1input = torch.rand([|10L|]).``to``(device)

let t' = m.forward t1input