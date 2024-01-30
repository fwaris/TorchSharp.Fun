module rec TorchSharp.Fun
open System
open TorchSharp
open type TorchSharp.torch.nn
open Microsoft.FSharp.Quotations

/// Pass arbitrary number of named arguments. 
/// Names beginning with underscore '_' are considered optional
/// type is inferred at access
type Args() =
    let d = System.Collections.Generic.Dictionary<string,obj>()
    member _.set name value = d.[name] <- box value
    member _.get name = d.[name]
    member _.tryGet name = match d.TryGetValue name with true,v -> Some v | _ -> None
    static member (?<-) (o:Args,name:string,value:'R) : Args = o.set name value; o

    ///access required argument
    static member inline (?) (o:Args,name:string) :^R = o.get name |> box :?> ^R

    ///access optional argument
    static member inline ( ?* ) (o:Args,name:string) :^R option =      
        let r = o.tryGet name 
        match r with 
        | Some j -> j :?> ^R |> Some
        | None   -> None            

type IModel =
    abstract forward : torch.Tensor->torch.Tensor
    abstract forward : torch.Tensor*Args -> torch.Tensor*Args //multiple inputs and outputs
    abstract Module : Module
    ///Hack to move buffers to the same device as the module - does not work if FuncModel is nested deep
    ///May need to manage buffers manually in some cases
    abstract to' : torch.Device -> unit

//let randName() = Guid.NewGuid().ToString()

let parseName (n:string) =
    let i = n.LastIndexOf("_")
    if i < 0 then
        n,0
    else
        let rs = n.Substring(i+1)
        let num =
            match Int32.TryParse rs with
            | true , x   -> x 
            | false, _   -> 0
        let prefix = n.Substring(0,i)
        prefix,num

let makeUnique name names =
    let rec makeUnique name names =
        match names with
        | [] -> name
        | x::rest when name = x -> 
            let n,num = parseName x
            let n2 = $"{n}_{num+1}"
            makeUnique n2 rest
        | _::rest -> makeUnique name rest
    makeUnique name (List.sort names)

let registerNamed (parent:#Module) (name,child:#Module) = 
    if child <> null then 
        let uname = parent.named_children() |> Seq.map(fun struct(n,_) -> n) |> Seq.toList |> makeUnique name 
        parent.register_module(uname,child)   
    
let internalMoveToDevice (m:torch.nn.Module) (device:torch.Device) =
    let submodules (m:torch.nn.Module) = m.named_modules() |> Seq.map(fun struct(_,m) -> m)
    let rec childModules acc (m:torch.nn.Module) =
        let acc = 
            match m with 
            | :? FuncModel as fm -> ((fm::acc),submodules fm) ||> Seq.fold childModules
            | m                  -> (acc,submodules m) ||> Seq.fold childModules
        acc
    let m = m.``to``(device)                                //first move module to device
    let cs = childModules [] m                              //find all FuncModel child instances
    cs |> Seq.iter (fun fm -> fm.FixBufferRefs(device))     //fix buffer 'ref's to point to new tensor on device

///convert object to IModel if convertible
let inline M< ^Q when ^Q : (member forward:torch.Tensor->torch.Tensor)>  (mdl:^Q) =
    match box mdl with
    | :? IModel as f ->  f
    | :? Module as m -> {new IModel with
                           member x.forward(t) = (^Q : (member forward:torch.Tensor->torch.Tensor)(mdl,t))
                           member x.forward(t,ts) = let t' = x.forward(t) in t',ts
                           member _.Module = m
                           member this.to' (device) = internalMoveToDevice this.Module device
                        }
    | x              -> failwith $"{x} is not convertible to IModel"

type Dependent = Md of Module | Im of IModel | Pr of Ref<Modules.Parameter> | Bu of Ref<torch.Tensor>
let toDep (o:obj) =
    match o with
    | :? Module as m -> Md m
    | :? IModel as m -> Im m
    | :? Ref<Modules.Parameter> as p -> Pr p
    | :? Ref<torch.Tensor> as t -> Bu t
    | x -> failwith $"{x} cannot be registered as dependent of a module. Valid dependents are torch.nn.Module, IModel, Ref<Modules.Parameter> and Ref<torch.Tensor>"


let genUniqueName key acc name = 
    let ns =
        match acc |> Map.tryFind key with
        | Some names -> let n = makeUnique name names in n::names
        | None       -> [name]
    acc |> Map.add key ns

let parmName (p:Modules.Parameter) = if String.IsNullOrWhiteSpace p.name then "parm" else p.name
let buffName (t:torch.Tensor) = if String.IsNullOrWhiteSpace t.name then "buff" else t.name
    
let genNames (deps:Dependent seq) =
    let nameMap =
        (Map.empty,deps)
        ||> Seq.fold (fun acc d -> 
            match d with
            | Md m -> genUniqueName 'M' acc (m.GetName()) 
            | Im m -> genUniqueName 'I' acc (m.Module.GetName())
            | Pr p -> genUniqueName 'P' acc (parmName p.Value)
            | Bu t -> genUniqueName 'T' acc (buffName t.Value)
            )
        |> Map.map(fun k v -> List.rev v)
    
    (([],nameMap),deps)
    ||> Seq.fold (fun (ls,nm) d -> 
        match d with
        | Md _ -> let ns = nm.['M'] in ((ns.Head,d)::ls),(nm |> Map.add 'M' ns.Tail)
        | Im _ -> let ns = nm.['I'] in ((ns.Head,d)::ls),(nm |> Map.add 'I' ns.Tail)
        | Pr _ -> let ns = nm.['P'] in ((ns.Head,d)::ls),(nm |> Map.add 'P' ns.Tail)
        | Bu _ -> let ns = nm.['T'] in ((ns.Head,d)::ls),(nm |> Map.add 'I' ns.Tail))
    |> fst


type torch.nn.Module with
    ///Hack to move buffers to the same device as the module 
    member this.to'(device:torch.Device) = internalMoveToDevice this device

///supporting class for creating 'forward' functions in a 'functional' way.
type FuncModel(name,dependents:(string*Dependent) seq,fwd:torch.Tensor->torch.Tensor, fwdExt:(torch.Tensor*Args->torch.Tensor*Args)) as this =
    inherit Module<torch.Tensor,torch.Tensor>(name)
    do
        dependents 
        |> Seq.iter (fun (n,d) ->
            match d with
            | Md m -> this.register_module(n,m)
            | Im m -> this.register_module(n,m.Module)
            | Pr p -> p.Value.name <- n; this.register_parameter(n,p.Value)
            | Bu t -> t.Value.name <- n; this.register_buffer(n,t.Value))

    override this.forward(t) = let t' = fwd t in t'
    member this.Module : Module = this :> _

    member internal this.FixBufferRefs(device:torch.Device) =
        let m = this.Module
        dependents
        |> Seq.iter (fun (n,d) ->
            match d with
            | Pr p -> p.Value <- m.get_parameter(p.Value.name); // update: seems parameters are moved properly now in torchsharp. just fix reference
            | Bu t -> t.Value <- m.get_buffer(t.Value.name); 
            | Im m -> ()
            | Md m -> ())
        

    //newer versions of PyTorch/TorchSharp require non-modules to be handled differently
    member this.to'(device:torch.Device) = 
        //let m = this.Module.``to``(device) 
        internalMoveToDevice this.Module device

    interface IModel with
        member this.forward(t) = this.forward(t)
        member this.forward(t,ts:Args) : (torch.Tensor * Args)= fwdExt(t,ts)
        member this.Module = this :> _ 
        member this.to'(device) = this.to'(device)

let extend (fwd:torch.Tensor->torch.Tensor) = fun (t,ts:Args) -> fwd t,ts
let notImplFwd (fwd:torch.Tensor) : torch.Tensor = failwith "Not implemented. Call with extended version of fwd that includes Args"

let checkNames names dependents =
    if Seq.length names <> Seq.length dependents + 1 then 
        failwithf $"number of names should be 1 + the-number-of-child-modules. The first name is for the module itself. Expecting {Seq.length dependents + 1} name(s) but got {Seq.length names}"

///Create a model (module) from the given function and register the childModules and parameters, if not empty
///If names is not empty, the function and its children will be assigned the given names. Count of names is 1 + number of childModules
let inline F (names:string seq) (dependents:obj seq) (fwd:torch.Tensor -> torch.Tensor) =
    if Seq.isEmpty names then
        let ds = dependents |> Seq.map toDep |> genNames
        let p = new FuncModel("funcModel",ds,fwd, extend fwd) 
        p :> IModel
    else
        checkNames names dependents
        let ds = dependents |> Seq.map toDep |> Seq.zip (Seq.tail names) 
        let p = new FuncModel(Seq.head names, ds, fwd, extend fwd) 
        p :> IModel

///Create a model (module) from the given function and register the childModules and parameters, if not empty
///If names is not empty, the function and its children will be assigned the given names. Count of names is 1 + number of childModules
///This version requires a second argument of type Args that supplies a list of named parameters
let inline Fx (names:string seq) (dependents:obj seq) fwd = 
    if Seq.isEmpty names then
        let ds = dependents |> Seq.map toDep |> genNames
        let p = new FuncModel("funcModel",ds,notImplFwd,fwd) 
        p :> IModel
    else
        checkNames names dependents
        let ds = dependents |> Seq.map toDep |> Seq.zip (Seq.tail names) 
        let p = new FuncModel(Seq.head names, ds, notImplFwd,fwd) 
        p :> IModel

let inline (=>>) m1 (n,m2) = 
    let m1 = M m1
    let m2 = M m2 
    registerNamed m1.Module (n,m2.Module)
    {new IModel with
        member _.forward(t) =
            use t' = m1.forward(t)
            m2.forward(t')
        member _.forward(t,ts) =
            let t',ts' = m1.forward(t,ts)
            let t2_s = m2.forward(t',ts')
            t'.Dispose()
            t2_s
        member _.Module = m1.Module
        member _.to'(device) = m1.to'(device)
    }

let inline (->>) m1 m2 =  
    let m1 = M m1
    let m2 = M m2
    m1 =>> (m2.Module.GetName(),m2)

module Tensor = 
    //Note: ensure 't matches tensor datatype otherwise ToArray might crash the app (i.e. exception cannot be caught)
    let inline private _getData<'t when 't: unmanaged and 't:>ValueType and 't:struct and 't : (new:unit->'t) > (t:torch.Tensor) =
        let s = t.data<'t>()
        s.ToArray()

    let getData<'t when 't: unmanaged and 't:>ValueType and 't:struct and 't : (new:unit->'t)>  (t:torch.Tensor) =
        if t.device_type <> DeviceType.CPU then 
            //use t1 = t.clone()
            use t2 = t.cpu()
            _getData<'t> t2
        else 
            _getData<'t> t
  
    let setData<'t when 't: unmanaged and 't:>ValueType and 't:struct and 't : (new:unit->'t)> (t:torch.Tensor) (data:'t[]) =
        if t.device_type = DeviceType.CPU |> not then failwith "tensor has to be on cpu for setData"        
        let s = t.data<'t>()
        s.CopyFrom(data,0,0L)

    type D<'a> = 
        | A of 'a[]         // flat array of values - from the inner most dimension
        | G of D<'a>[]      // group of groups or flat arrays

    //utility function to get raw tensor data as a recursive structure for debugging purposes
    let getDataNested<'a when 'a: unmanaged and  'a: (new: unit -> 'a) and  'a: struct and 'a :> ValueType>(t:torch.Tensor) = 
        let ts = if t.device<>torch.CPU then t.cpu().data<'a>().ToArray() else t.data<'a>().ToArray()
        let rdims =
            t.shape 
            |> Array.map int 
            |> Array.rev            //start with inner most dimension
            |> Array.toList
        let rec loop ds (xs:D<'a>) =
            match ds,xs with
            | [],_                        -> xs
            | d::[],G ds when d=ds.Length -> G ds
            | d::[],A ds when d=ds.Length -> A ds
            | d::rest,G ds -> loop rest (ds |> Array.chunkBySize d |> Array.map G |> G)
            | d::rest,A ds -> loop rest (ds |> Array.chunkBySize d |> Array.map A |> G)
        loop rdims (A ts)



