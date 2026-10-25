"""Real offline Transformer and pooling checks for the E5 inference path."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from research.generative_recsys.src.tokenizer import content_embeddings, validate_prepared_config


@unittest.skipUnless(importlib.util.find_spec('transformers'), 'Transformers not installed')
class E5Tests(unittest.TestCase):
    def test_local_encoder_pooling_prefix_and_complete_cpu_pipeline(self):
        import torch
        from transformers import BertConfig, BertModel, BertTokenizerFast
        torch.set_num_threads(2)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            vocab = root/'vocab.txt'
            vocab.write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]',
                                      'query',':','фильм','драма','комедия']))
            tokenizer = BertTokenizerFast(vocab_file=str(vocab),do_lower_case=True)
            model = BertModel(BertConfig(vocab_size=10,hidden_size=16,num_hidden_layers=1,
                num_attention_heads=2,intermediate_size=32)).eval()
            checkpoint = root/'encoder'; model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            config = json.loads(Path('research/generative_recsys/configs/default.json').read_text())
            config.update(embedding_model=str(checkpoint),embedding_dim=16,embedding_batch_size=2)
            texts = ['фильм','фильм драма комедия']
            actual, metadata = content_embeddings(texts,config,'cpu')
            inputs = tokenizer(['query: '+text for text in texts],padding=True,return_tensors='pt')
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                mask = inputs.attention_mask.unsqueeze(-1)
                pooled = (hidden*mask).sum(1)/mask.sum(1)
                expected = torch.nn.functional.normalize(pooled,dim=1).numpy()
            np.testing.assert_allclose(actual,expected,atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(actual,axis=1),1,atol=1e-6)
            self.assertEqual(metadata['prefix'],'query: ')
            self.assertEqual(metadata['max_length'],256)
            config_file = root/'config.json'; config_file.write_text(json.dumps(config))
            result = subprocess.run([sys.executable,'-m','research.generative_recsys.run',
                '--smoke','--pretrained-smoke','--config',str(config_file),'--output',str(root/'run')],
                capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(json.loads((root/'run/status.json').read_text())['status'],'completed')
            for changes in [{'embedding_backend':'sentence_t5'}, {'embedding_max_length':128}]:
                with self.assertRaises(ValueError):
                    validate_prepared_config(config,{**config,**changes})
