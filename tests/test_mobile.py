"""Real ONNX classifier export, mobile manifest integrity and fail-closed guards."""
from dataclasses import asdict,replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from sokolov.config import load_config
from sokolov.io import sha256,write_json,read_json

ROOT=Path(__file__).resolve().parents[1]


class MobileExportTests(unittest.TestCase):
    def test_both_best_checkpoints_export_with_parity_and_technical_guard(self):
        import torch
        import numpy as np
        import onnx
        from onnx import helper,TensorProto,numpy_helper
        from sokolov.model import build_model
        from sokolov.mobile import export_mobile
        torch.set_num_threads(2)
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            detector=root/'det.onnx'
            nodes=[]; outputs=[]
            for i in range(9):
                stride=(8,16,32)[i%3]; width=(1,4,10)[i//3]
                data=np.zeros(((640//stride)**2*2,width),dtype=np.float32)
                nodes.append(helper.make_node('Constant',[],[f'out{i}'],value=numpy_helper.from_array(data)))
                outputs.append(helper.make_tensor_value_info(f'out{i}',TensorProto.FLOAT,list(data.shape)))
            graph=helper.make_graph(nodes,'fixture-SCRFD',[helper.make_tensor_value_info('pixels',TensorProto.FLOAT,[1,3,640,640])],outputs)
            onnx.save(helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=9),detector)
            write_json(root/'data/data_audit.json',{'detector':{'sha256':sha256(detector)}})
            for name in ('resnet18','arcface'):
                config=load_config(ROOT/f'configs/{name}_face.toml')
                config=replace(config,training=replace(config.training,device='cpu'))
                model=build_model(config.model,pretrained=False).eval()
                folder=root/name;folder.mkdir()
                torch.save({'model':model.state_dict(),'config':asdict(config),'technical_test':True,
                    'data_fingerprint':{'shared':'fixture'},'best_ap':.8,'epoch':0},folder/'best.pt')
                write_json(folder/'status.json',{'status':'evaluated'})
                write_json(folder/'threshold.json',{'selection_partition':'val','threshold':.63,
                    'checkpoint_sha256':sha256(folder/'best.pt')})
                del model
            with self.assertRaisesRegex(ValueError,'technical'):
                export_mobile(root,root/'forbidden.zip',detector=detector)
            output=export_mobile(root,root/'test-mobile.zip',detector=detector,allow_technical=True)
            manifest=read_json(output.with_suffix('.json'))
            self.assertTrue(manifest['technical_test'])
            self.assertEqual([m['input_size'] for m in manifest['models']],[224,112])
            self.assertTrue(all(m['max_parity_error']<=1e-4 for m in manifest['models']))
            self.assertTrue(all(m['threshold']==.63 for m in manifest['models']))
            with zipfile.ZipFile(output) as z:
                self.assertEqual(set(z.namelist()),{'manifest.json','detector.onnx','resnet18.onnx','arcface.onnx'})
                z.extractall(root/'unpacked')
            for m in manifest['models']:
                self.assertEqual(sha256(root/'unpacked'/m['file']),m['sha256'])
            # No production bundle can be exported with a threshold from another checkpoint.
            threshold=read_json(root/'resnet18/threshold.json'); threshold['checkpoint_sha256']='wrong'
            write_json(root/'resnet18/threshold.json',threshold)
            with self.assertRaisesRegex(ValueError,'threshold'):
                export_mobile(root,root/'wrong.zip',detector=detector,allow_technical=True)
