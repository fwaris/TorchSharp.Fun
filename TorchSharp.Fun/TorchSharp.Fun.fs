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
    static member inline (?) (o:Args,name:string) :^R = (if name.StartsWith("_") then  o.tryGet name |> box else o.get name |> box) :?> ^R
    static member (?<-) (o:Args,name:string,value:'R) : Args = o.set name value; o
    

type IModel =
    abstract forward : torch.Tensor->torch.Tensor
    abstract forward : torch.Tensor*Args -> torch.Tensor*Args //multiple inputs and outputs
    abstract Module : Module

let randName() = Guid.NewGuid().ToString()

let registerNamed (parent:#Module) (name,child:#Module) = if child <> null then parent.register_module(name,child)    

let register (parent:IModel) (child:IModel) =
    registerNamed parent.Module (randName(),child.Module)

///convert object to IModel if convertible
let inline M< ^Q when ^Q : (member forward:torch.Tensor->torch.Tensor)>  (mdl:^Q) =
    match box mdl with
    | :? IModel as f ->  f
    | :? Module as m -> {new IModel with
                           member x.forward(t) = (^Q : (member forward:torch.Tensor->torch.Tensor)(mdl,t))
                           member x.forward(t,ts) = let t' = x.forward(t) in t',ts
                           member _.Module = m
                        }
    | x              -> failwith $"{x} is not convertible to IModel"

let registerNamedChildren names (childModules:IModel seq) parent =
    Seq.zip (Seq.tail names) childModules     
    |> Seq.iter (fun (n,c) -> registerNamed parent (n,c.Module))

let registerChildren children parent =
    children |> Seq.iter (fun c -> register parent c)

[<AbstractClass>]
type AbstractModel() =
    abstract member forward : torch.Tensor -> torch.Tensor
    abstract member Module : Module    
    member this.forward (t,ts) = let t' = this.forward(t) in t',ts

    interface IModel with
        member this.forward t = this.forward(t)
        member this.forward(x,ts) = this.forward(x,ts)
        member this.Module = this.Module
    //operators TBD
    static member (+) (a:IModel,b:torch.Tensor) = {new IModel with
                                                        member _.forward(t) = use t' = a.forward t in t' + b
                                                        member this.forward(t,ts) =    
                                                            let t' = this.forward(t)
                                                            t',ts
                                                        member _.Module = a.Module
                                                 }

type FuncModel(name,parameters:Modules.Parameter[],fwd:torch.Tensor->torch.Tensor) as this =
    inherit Module(name)
    do parameters |> Array.iter (fun p -> this.register_parameter(p.name,p.t(),p.requires_grad))
    override this.forward(t) = let t' = fwd t in t'
    member this.Module : Module = this :> _

    interface IModel with
        member this.forward(t) = this.forward(t)
        member this.forward(t,ts) = 
            let t1 =  this.forward(t)
            t1,ts
        member this.Module = this :> _ 

type FuncModelList(name,parameters:Modules.Parameter[],fwd) as this =
    inherit Module(name)
    do parameters |> Array.iter (fun p -> this.register_parameter(p.name,p.t(),p.requires_grad))
    override this.forward(t) = fwd(t,Args()) |> fst
    member this.Module : Module = this :> _
    interface IModel with
        member this.forward(t) = this.forward(t)
        member this.forward(t,ts) = fwd(t,ts)
        member this.Module = this :> _    

    
///Create a model (module) from the given function and register the childModules as children
let inline F childModules (fwd:torch.Tensor -> torch.Tensor) =
    let p = new FuncModel(randName(), [||],fwd) 
    registerChildren childModules p
    p

///Create a model (module) from the given function. Register the childModules as children and add the parameters to the model
let inline  F' childModules (parameters:Modules.Parameter seq) (fwd:torch.Tensor -> torch.Tensor) =
    let p = new FuncModel(randName(), Seq.toArray parameters,fwd) 
    registerChildren childModules p
    p

///Create a model (module) from the given function and register the childModules as children
let inline Fl childModules fwd =
    let p = new FuncModelList(randName(), [||],fwd) :> IModel
    registerChildren childModules p
    p

///Create a model (module) from the given function. Register the childModules as children and add the parameters to the model
let inline  Fl' childModules (parameters:Modules.Parameter seq) fwd =
    let p = new FuncModelList(randName(), Seq.toArray parameters,fwd) 
    registerChildren childModules p
    p

let checkNames names childModules =
    if Seq.length names <> Seq.length childModules + 1 then 
        failwithf $"number of names should be 1 + the-number-of-child-modules. The first name is for the module itself. Expecting {Seq.length childModules + 1} name(s) but got {Seq.length names}"

///<summary>Same as F but now assign names to all models (modules)</summary>
/// <seealso cref="F" />
let inline Fn names childModules (fwd:torch.Tensor -> torch.Tensor) =
    checkNames names childModules
    let p = new FuncModel(Seq.head names, [||],fwd) 
    registerNamedChildren names childModules p
    p

///<summary>Same as F' but now assign names to all models (modules)</summary>
/// <seealso cref="F'" />
let inline Fn' names childModules (parameters:Modules.Parameter seq) (fwd:torch.Tensor -> torch.Tensor) =
    checkNames names childModules
    let p = new FuncModel(Seq.head names, Seq.toArray parameters,fwd) 
    registerNamedChildren names childModules p
    p

///<summary>Same as F but now assign names to all models (modules)</summary>
/// <seealso cref="F" />
let inline Fnl names childModules fwd =
    checkNames names childModules
    let p = new FuncModelList(Seq.head names, [||],fwd) 
    registerNamedChildren names childModules p
    p

///<summary>Same as F' but now assign names to all models (modules)</summary>
/// <seealso cref="F'" />
let inline Fnl' names childModules (parameters:Modules.Parameter seq) fwd =
    checkNames names childModules
    let p = new FuncModelList(Seq.head names, Seq.toArray parameters,fwd) 
    registerNamedChildren names childModules p
    p

let inline (->>) m1 m2 = 
    let m1 = M m1
    let m2 = M m2 
    registerNamed m1.Module ($"{m2.Module.GetName()}_{randName()}",m2.Module)
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
    }
//let inline (=>>) m1 (n,m2) = compose m1 (Some n, m2)

module Tensor = 
    //Note: ensure 't matches tensor datatype otherwise ToArray might crash the app (i.e. exception cannot be caught)
    let private _getData<'t when 't:>ValueType and 't:struct and 't : (new:unit->'t) > (t:torch.Tensor) =
        let s = t.data<'t>()
        s.ToArray()

    let getData<'t when 't:>ValueType and 't:struct and 't : (new:unit->'t)>  (t:torch.Tensor) =
        if t.device_type <> DeviceType.CPU then 
            //use t1 = t.clone()
            use t2 = t.cpu()
            _getData<'t> t2
        else 
            _getData<'t> t
  
    let setData<'t when 't:>ValueType and 't:struct and 't : (new:unit->'t)> (t:torch.Tensor) (data:'t[]) =
        if t.device_type = DeviceType.CPU |> not then failwith "tensor has to be on cpu for setData"        
        let s = t.data<'t>()
        s.CopyFrom(data,0,0L)

module Model =
    open MBrace.FsPickler
    open Tensor

   // Enum.GetNames(typeof<torch.ScalarType>)
    type TnsrData =
        | FByte of byte[]
        | FInt8 of int8[]
        | FInt16 of int16[]
        | FInt32 of int32[]
        | FInt64 of int64[]
        | FFloat32 of float32[]
        | FFloat64 of float[]
        | FBool of bool[]
        | FBFFloat16 of Half[]

    let getTnsrData (t:torch.Tensor) =
        match t.dtype with 
        | torch.ScalarType.Byte     -> getData<byte> t      |> FByte
        | torch.ScalarType.Int8     -> getData<int8> t      |> FInt8
        | torch.ScalarType.Int16    -> getData<int16> t     |> FInt16
        | torch.ScalarType.Int32    -> getData<int> t       |> FInt32
        | torch.ScalarType.Int64    -> getData<int64> t     |> FInt64
        | torch.ScalarType.Float32  -> getData<float32> t   |> FFloat32
        | torch.ScalarType.Float64  -> getData<float> t     |> FFloat64
        | torch.ScalarType.BFloat16 -> getData<Half> t      |> FBFFloat16
        | x -> failwith $"data type %A{x} not handled"

    let setTnsrData td (t:torch.Tensor) =
        match td with
        | FByte ds      -> setData<byte> t ds
        | FInt8 ds      -> setData<int8> t ds
        | FInt16 ds     -> setData<int16> t ds
        | FInt32 ds     -> setData<int> t ds
        | FInt64 ds     -> setData<int64> t ds
        | FFloat32 ds   -> setData<float32> t ds
        | FFloat64 ds   -> setData<float> t ds
        | FBool ds      -> setData<bool> t ds
        | FBFFloat16 ds -> setData<Half> t ds

    let saveParms file (model:IModel) =
        let parms = model.Module.parameters()
        let values = parms |> Array.map getTnsrData
        let ser = FsPickler.CreateBinarySerializer()
        use str = IO.File.Create(file:string)
        ser.Serialize(str,values)

    let loadParms file (model:IModel) =
        let ser = FsPickler.CreateBinarySerializer()
        use str = IO.File.OpenRead(file:string)
        let values = ser.Deserialize<TnsrData[]>(str)
        let parms = model.Module.parameters()
        Array.zip values parms
        |> Array.iter (fun (td,t) -> setTnsrData td t)

    let dipsose (m:IModel) = if m.Module <> null then m.Module.Dispose()

// module private _check_ = 
//     open type TorchSharp.NN.Modules
//     open TorchSharp.Fun

//     let s1 = SELU()
//     let s1n = Linear(1L,1L)
//     let s2 = GELU()
    
//     let m1 = s1 ->> s1n
//     let m3 = s2 ->> m1


