"""Create tiny deterministic ONNX fixtures, exclusively for Android runtime tests.

These graphs are NOT face recognition models. They exercise ZIP/ORT/CameraX/UI
plumbing, use synthetic detections/scores, and always carry technical_test=true.
"""
from pathlib import Path
import hashlib,json,zipfile,tempfile
import numpy as np
import onnx
from onnx import helper,TensorProto,numpy_helper

root=Path(__file__).resolve().parents[1]/'work/android-test-assets'
root.mkdir(parents=True,exist_ok=True)
def save_graph(nodes,inputs,outputs,path):
    onnx.save(helper.make_model(helper.make_graph(nodes,'TECHNICAL_FIXTURE',inputs,outputs),
        opset_imports=[helper.make_opsetid('',17)],ir_version=9),path)
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory() as temporary:
    folder=Path(temporary)
    models=[]
    for name,size,score in [('resnet18',224,.9),('arcface',112,.1)]:
        path=folder/f'{name}.onnx'
        save_graph([helper.make_node('Constant',[],['score'],value=numpy_helper.from_array(np.array([[score]],np.float32)))],
            [helper.make_tensor_value_info('pixels',TensorProto.FLOAT,[1,3,size,size])],
            [helper.make_tensor_value_info('score',TensorProto.FLOAT,[1,1])],path)
        models.append(dict(id=name,file=path.name,sha256=digest(path),input_size=size,
            input_name='pixels',output_name='score',threshold=.5))
    nodes=[helper.make_node('ReduceMean',['pixels'],['mean'],keepdims=0),
        helper.make_node('Constant',[],['limit'],value=numpy_helper.from_array(np.array(-.8,np.float32))),
        helper.make_node('Greater',['mean','limit'],['present']),
        helper.make_node('Cast',['present'],['factor'],to=TensorProto.FLOAT)]
    outputs=[]
    for i in range(9):
        stride=(8,16,32)[i%3]; width=(1,4,10)[i//3]
        values=np.zeros(((640//stride)**2*2,width),np.float32)
        anchor=2*(40*80+40)
        if i==0: values[anchor]=.99
        if i==3: values[anchor]=[10,12,10,10]
        if i==6: values[anchor]=[-5,-4.375,5,-4.375,0,1.875,-3.75,6.875,3.75,6.875]
        nodes.append(helper.make_node('Constant',[],[f'constant{i}'],value=numpy_helper.from_array(values)))
        nodes.append(helper.make_node('Mul',[f'constant{i}','factor'],[f'out{i}']))
        outputs.append(helper.make_tensor_value_info(f'out{i}',TensorProto.FLOAT,list(values.shape)))
    path=folder/'detector.onnx'
    save_graph(nodes,[helper.make_tensor_value_info('pixels',TensorProto.FLOAT,[1,3,640,640])],outputs,path)
    manifest=dict(schema_version=1,technical_test=True,default_model='resnet18',
        class_names=['not_sokolov','sokolov'],models=models,
        detector=dict(file=path.name,sha256=digest(path),input_size=640,threshold=.5,nms_threshold=.4))
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with zipfile.ZipFile(root/'fixture_models.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.iterdir()):archive.write(path,path.name)
print(root/'fixture_models.zip')
