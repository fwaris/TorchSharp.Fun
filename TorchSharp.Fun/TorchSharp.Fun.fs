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

let rec makeUnique name names =
    match names with
    | [] -> name
    | x::rest when name = x -> 
        let n,num = parseName x
        let n2 = $"{n}_{num+1}"
        makeUnique n2 rest
    | _::rest -> makeUnique name rest

let registerNamed (parent:#Module) (name,child:#Module) = 
    if child <> null then 
        let uname = parent.named_children() |> Seq.map(fun struct(n,_) -> n) |> Seq.toList |> makeUnique name 
        parent.register_module(uname,child)    

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

let inline registerNamedChildren names (childModules:IModel seq) parent =
    let parent = M parent
    Seq.zip (Seq.tail names) childModules     
    |> Seq.iter (fun (n,c) -> registerNamed parent.Module (n,c.Module))

let inline registerChildren children parent =
    let names = Seq.append ["dummy"] (children |> Seq.map (fun c -> (M c).Module.GetName()))
    registerNamedChildren names children parent

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
///supporting class for creating 'forward' functions in a 'functional' way.
type FuncModel(name,parameters:Modules.Parameter[],fwd:torch.Tensor->torch.Tensor, fwdExt:(torch.Tensor*Args->torch.Tensor*Args)) as this =
    inherit Module(name)
    do parameters |> Array.iter (fun p -> this.register_parameter(p.name,p))
    override this.forward(t) = let t' = fwd t in t'
    member this.Module : Module = this :> _

    interface IModel with
        member this.forward(t) = this.forward(t)
        member this.forward(t,ts:Args) : (torch.Tensor * Args)= fwdExt(t,ts)
        member this.Module = this :> _ 

let extend (fwd:torch.Tensor->torch.Tensor) = fun (t,ts:Args) -> fwd t,ts
let notImplFwd (fwd:torch.Tensor) : torch.Tensor = failwith "Not implemented. Call with extended version of fwd that includes Args"

let checkNames names childModules =
    if Seq.length names <> Seq.length childModules + 1 then 
        failwithf $"number of names should be 1 + the-number-of-child-modules. The first name is for the module itself. Expecting {Seq.length childModules + 1} name(s) but got {Seq.length names}"

///Create a model (module) from the given function and register the childModules and parameters, if not empty
///If names is not empty, the function and its children will be assigned the given names. Count of names is 1 + number of childModules
let inline F (names:string seq) childModules (parameters:Modules.Parameter seq) (fwd:torch.Tensor -> torch.Tensor) =
    if Seq.isEmpty names then
        let p = new FuncModel("funcModel", Seq.toArray parameters,fwd, extend fwd) 
        registerChildren childModules p
        p :> IModel
    else
        checkNames names childModules
        let p = new FuncModel(Seq.head names, Seq.toArray parameters, fwd, extend fwd) 
        registerNamedChildren (Seq.tail names) childModules p
        p :> IModel

///Create a model (module) from the given function and register the childModules and parameters, if not empty
///If names is not empty, the function and its children will be assigned the given names. Count of names is 1 + number of childModules
///This version requires a second argument of type Args that supplies a list of named parameters
let inline Fx (names:string seq) childModules (parameters:Modules.Parameter seq) fwd = 
    if Seq.isEmpty names then
        let p = new FuncModel("funcModel", Seq.toArray parameters,notImplFwd, fwd) 
        registerChildren childModules p
        p :> IModel
    else
        checkNames names childModules
        let p = new FuncModel(Seq.head names, Seq.toArray parameters, notImplFwd, fwd) 
        registerNamedChildren (Seq.tail names) childModules p
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
    }

let inline (->>) m1 m2 =  
    let m1 = M m1
    let m2 = M m2
    m1 =>> (m2.Module.GetName(),m2)

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

    type D<'a> = 
        | A of 'a[]         // flat array of values - from the inner most dimension
        | G of D<'a>[]      // group of groups or flat arrays

    //utility function to get raw tensor data as a recursive structure for debugging purposes
    let getDataNested<'a when 'a: (new: unit -> 'a) and  'a: struct and 'a :> ValueType>(t:torch.Tensor) = 
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

module Model =
    open MBrace.FsPickler
    open Tensor

    let saveParms file (model:IModel) = model.Module.save(file:string) |> ignore

    let loadParms file (model:IModel) = model.Module.load(file:string) |> ignore

    let dipsose (m:IModel) = if m.Module <> null then m.Module.Dispose()

    (* older method of saving model - not used
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
    *)
// module private _check_ = 
//     open type TorchSharp.NN.Modules
//     open TorchSharp.Fun

//     let s1 = SELU()
//     let s1n = Linear(1L,1L)
//     let s2 = GELU()
    
//     let m1 = s1 ->> s1n
//     let m3 = s2 ->> m1



