#r "nuget: TorchSharp-cuda-windows" 
open TorchSharp

type M1(n) as this =
   inherit torch.nn.Module<torch.Tensor,torch.Tensor>(n)
   let buff = torch.ones([|1L;5L;|])
   let linear = torch.nn.Linear(10L,5L)

   do this.RegisterComponents()

   override _.forward t = (t --> linear) + buff

let device = if torch.cuda_is_available() then torch.CUDA else failwith "this test needs cuda"

let m1 = (new M1("test")).``to``(device)

let tinp = torch.rand([|1L; 10L;|], device=device)
let tout = tinp --> m1


